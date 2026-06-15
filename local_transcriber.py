"""
Local Transcriber.

This is a small desktop app that turns audio and video into text transcripts. It
runs on your own computer. After the first run downloads a model one time, it does
not need the internet, and it does not use any paid service.

It uses faster-whisper, which basically means a faster and lighter version of
OpenAI's Whisper speech to text model. faster-whisper reads audio and video on its
own, so you do not need a separate ffmpeg install.

It runs on your NVIDIA card when one is available, and it falls back to the
processor when one is not. On the card it uses float16, which is just a compact
number format. It does not use int8 on the card on purpose. The 50 series cards use
NVIDIA's newest architecture, called Blackwell, and int8 currently crashes on
Blackwell cards with a cuBLAS error. float16 is the safe setting. On the processor
it uses int8, which is fine there.
"""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, scrolledtext, ttk


# These are the file types the app treats as input. The watch folder only reacts to
# these, so the text files and subtitle files the app writes are never picked up as
# new input.
AUDIO_VIDEO_EXTENSIONS = {
    # audio
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
    # video
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".3gp",
}

# The models you can pick in the dropdown. The default is large-v2, which held up
# better than the newer large-v3 on classroom and lecture audio in the testing this
# app was built around. The others are there so you can try them when you want.
MODEL_CHOICES = ["large-v2", "large-v3", "distil-large-v3", "small.en"]
DEFAULT_MODEL = "large-v2"

# The app saves a transcript next to the source file, with this added to the name.
# So lecture1.mp4 produces lecture1_transcript.txt and lecture1_transcript.srt.
TRANSCRIPT_SUFFIX = "_transcript"

# The full explanation that shows up when you click the link next to the hints box.
# It lives here as one block so the window itself can stay short and uncluttered.
HINTS_HELP_TEXT = (
    "Whatever you type in the hints box is fed to the model as a starting prompt, "
    "which basically means the model reads it first, as if it were the words spoken "
    "right before your audio began. It is not added to your transcript. It only sets "
    "what the model expects to hear.\n"
    "\n"
    "This helps most with spelling. The model guesses spellings from sound, so "
    "uncommon names and subject words often come out wrong. If you give it the "
    "spelling up front, it tends to reuse it.\n"
    "\n"
    "Good things to put here:\n"
    "\n"
    "   •  The teacher's name.\n"
    "   •  Subject words, proper nouns, and acronyms for the lecture.\n"
    "\n"
    "For example:\n"
    "\n"
    "   Dr. Nguyen. AP Biology: mitochondria, endoplasmic reticulum, photosynthesis, "
    "Krebs cycle.\n"
    "\n"
    "A few limits worth knowing:\n"
    "\n"
    "   •  It is a nudge, not a command. It raises the odds of the right spelling, "
    "but it does not guarantee it.\n"
    "   •  It will not add words that were not actually said.\n"
    "   •  Keep it short. The model only reads the tail end of a long prompt, so a "
    "wall of text does not help.\n"
    "   •  Do not write instructions like 'transcribe this accurately'. The text is "
    "treated as example wording, not as a request.\n"
    "\n"
    "If you have no names or odd words for a file, just leave it empty. It is there "
    "for the cases where spelling matters to you.\n"
)


def add_cuda_dll_directories():
    """Tell Windows where the CUDA libraries live so Python can load them.

    The CUDA 12 libraries that faster-whisper needs (cuBLAS and cuDNN) install as
    pip packages under a folder called nvidia inside site-packages. On Windows,
    Python and CTranslate2 have to be told where those DLL files are before they can
    load them. CTranslate2 is the engine under faster-whisper, and it loads cuBLAS
    and cuDNN the moment it first uses the card.

    This finds each nvidia bin folder and makes it findable in two ways. It calls
    os.add_dll_directory, which covers libraries that load as a dependency of another
    library. It also puts the folder on PATH, which covers the libraries that
    CTranslate2 loads by name on its own. Both are needed, because add_dll_directory
    on its own does not catch that second case. The result is that the graphics card
    path works without you editing PATH by hand.

    This only runs on Windows and only adds folders that actually exist, so it is
    safe to call even when the CUDA libraries are not installed.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    bin_dirs = []
    for entry in list(sys.path):
        if not entry:
            continue
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in sorted(nvidia_root.glob("*/bin")):
            bin_dirs.append(str(bin_dir))
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            # If a folder cannot be added this way, skip it. The PATH addition below
            # still gives CTranslate2 a way to find the library, and the model load
            # further on will report a clear message if a library is genuinely
            # missing.
            pass
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


def detect_device():
    """Decide whether to run on the graphics card or the processor.

    The app asks CTranslate2, which is the engine under faster-whisper, whether an
    NVIDIA card is present. If a card is there, the app runs on it with float16. If
    no card is there, the app runs on the processor with int8.

    float16 and int8 are both just number formats for the model. float16 is the
    safe one for Blackwell cards, as explained at the top of this file. int8 is
    fine on the processor and keeps it quick enough to be usable.
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        # If CTranslate2 cannot be imported or cannot count devices, fall back to
        # the processor. The processor path does not need the CUDA libraries.
        pass
    return "cpu", "int8"


def format_timestamp_txt(seconds):
    """Turn a number of seconds into HH:MM:SS for the text transcript.

    The point of the timestamp is so you can scrub back to a spot in the video later
    to check a quote, so whole seconds are enough here.
    """
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_srt(seconds):
    """Turn a number of seconds into HH:MM:SS,mmm for the subtitle file.

    The subtitle format (.srt) wants milliseconds after a comma, so this keeps the
    fractional part instead of dropping it. A video player uses these times to show
    each line at the right moment.
    """
    if seconds is None or seconds < 0:
        seconds = 0
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def transcript_txt_path(source_path):
    source = Path(source_path)
    return source.with_name(f"{source.stem}{TRANSCRIPT_SUFFIX}.txt")


def transcript_srt_path(source_path):
    source = Path(source_path)
    return source.with_name(f"{source.stem}{TRANSCRIPT_SUFFIX}.srt")


def has_transcript(source_path):
    """Return True when a text transcript already sits next to the source file.

    Watch mode uses this to skip files it has already done, so it does not redo the
    whole folder every time.
    """
    return transcript_txt_path(source_path).exists()


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Transcriber")

        # Jobs waiting to be transcribed. Both the manual Transcribe button and the
        # watch folder feed this same queue, and a single worker takes one job at a
        # time. One at a time matters because the model holds the graphics card, and
        # two transcriptions at once would fight over it.
        self.job_queue = queue.Queue()

        # Messages from the worker and watch threads to the window. Tkinter is not
        # safe to touch from other threads, so those threads never change widgets
        # directly. They put messages here, and the main thread applies them.
        self.ui_queue = queue.Queue()

        # Work out the graphics card or processor up front so the window can show it.
        self.device, self.compute_type = detect_device()

        # The loaded model is reused between files. It is only reloaded when you pick
        # a different model in the dropdown.
        self.model = None
        self.loaded_model_key = None  # (model_name, device, compute_type)

        # Plain copies of the dropdown and hints box, refreshed on the main thread.
        # The background threads read these instead of touching the widgets.
        self.current_model = DEFAULT_MODEL
        self.current_hints = ""

        self.selected_file = None
        self.last_output_dir = None
        self.hints_help_window = None

        # Watch mode state.
        self.watch_dir = None
        self.watch_thread = None
        self.watch_stop = threading.Event()
        # path -> (size, time the file was first seen at this size). This is how the
        # app waits for a file to finish copying before it grabs it.
        self.seen_sizes = {}
        # Files already queued or already done this session, so they are not picked
        # up twice.
        self.handled = set()

        self.build_ui()

        # Start the single worker thread. It is a daemon thread, which basically
        # means it shuts down with the window instead of keeping the program alive.
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # Begin draining messages from the background threads.
        self.poll_ui_queue()

    # ----- building the window -------------------------------------------------

    def build_ui(self):
        pad = {"padx": 8, "pady": 5}

        buttons = ttk.Frame(self.root)
        buttons.pack(fill="x", **pad)

        self.select_button = ttk.Button(buttons, text="Select file", command=self.on_select_file)
        self.select_button.pack(side="left", padx=(0, 6))

        self.transcribe_button = ttk.Button(buttons, text="Transcribe", command=self.on_transcribe_clicked)
        self.transcribe_button.pack(side="left", padx=(0, 6))

        self.watch_button = ttk.Button(buttons, text="Watch folder", command=self.on_watch_clicked)
        self.watch_button.pack(side="left", padx=(0, 6))

        self.show_button = ttk.Button(buttons, text="Show folder", command=self.on_show_folder)
        self.show_button.pack(side="left", padx=(0, 6))

        model_row = ttk.Frame(self.root)
        model_row.pack(fill="x", **pad)
        ttk.Label(model_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=MODEL_CHOICES,
            state="readonly",
            width=18,
        )
        self.model_combo.pack(side="left", padx=(4, 0))

        hints_row = ttk.Frame(self.root)
        hints_row.pack(fill="x", **pad)

        # A short title with a blue link next to it. The link opens a popup with the
        # full description, so the window stays clean for people who already know
        # what the box is for.
        hints_header = ttk.Frame(hints_row)
        hints_header.pack(fill="x")
        ttk.Label(hints_header, text="Transcriber hints (optional)").pack(side="left")

        link_font = tkfont.Font(font="TkDefaultFont")
        link_font.configure(underline=True)
        hints_link = tk.Label(
            hints_header,
            text="What should I put here?",
            foreground="blue",
            cursor="hand2",
            font=link_font,
        )
        hints_link.pack(side="left", padx=(8, 0))
        hints_link.bind("<Button-1>", lambda event: self.show_hints_help())

        self.hints_text = tk.Text(hints_row, height=3, wrap="word")
        self.hints_text.pack(fill="x")

        self.file_label = ttk.Label(self.root, text="No file selected.")
        self.file_label.pack(fill="x", **pad)

        # The progress bar runs from 0 to 100 and follows how far into the audio the
        # transcription has reached.
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", **pad)

        self.status_var = tk.StringVar(value="Ready. Pick a file, or start Watch folder.")
        ttk.Label(self.root, textvariable=self.status_var).pack(fill="x", **pad)

        # A line that shows the graphics card or processor, so you can confirm at a
        # glance which one is in use.
        if self.device == "cuda":
            device_text = "Device: graphics card (cuda, float16)."
        else:
            device_text = "Device: processor (cpu, int8)."
        self.device_var = tk.StringVar(value=device_text)
        ttk.Label(self.root, textvariable=self.device_var).pack(fill="x", **pad)

        self.watch_var = tk.StringVar(value="Watch mode is off.")
        ttk.Label(self.root, textvariable=self.watch_var).pack(fill="x", **pad)

        # The transcript shows up here as it is produced.
        self.output = scrolledtext.ScrolledText(self.root, wrap="word", height=18)
        self.output.pack(fill="both", expand=True, **pad)

    # ----- button handlers (these run on the main thread) ----------------------

    def on_select_file(self):
        patterns = " ".join(f"*{ext}" for ext in sorted(AUDIO_VIDEO_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Pick an audio or video file",
            filetypes=[("Audio and video", patterns), ("All files", "*.*")],
        )
        if path:
            self.selected_file = path
            self.file_label.config(text=f"Selected: {path}")
            self.status_var.set("File selected. Press Transcribe when you are ready.")

    def on_transcribe_clicked(self):
        if not self.selected_file:
            self.status_var.set("Pick a file first with Select file.")
            return
        # Refresh the plain copies of the inputs from the widgets before queuing.
        self.current_model = self.model_var.get()
        self.current_hints = self.hints_text.get("1.0", "end").strip()
        self.enqueue_job(self.selected_file)

    def on_show_folder(self):
        folder = self.last_output_dir
        if not folder:
            self.status_var.set("There is no saved transcript yet.")
            return
        try:
            # On Windows this opens the folder in File Explorer.
            os.startfile(folder)
        except AttributeError:
            # A fallback for the rare case this is not Windows.
            subprocess.Popen(["explorer", folder])
        except OSError as error:
            self.status_var.set(f"Could not open the folder: {error}")

    def show_hints_help(self):
        # If the popup is already open, bring it to the front instead of opening a
        # second copy of it.
        if self.hints_help_window is not None and self.hints_help_window.winfo_exists():
            self.hints_help_window.lift()
            self.hints_help_window.focus_set()
            return

        window = tk.Toplevel(self.root)
        self.hints_help_window = window
        window.title("Transcriber hints")
        window.geometry("520x470")
        # Tie the popup to the main window, so it stays above it and closes with it.
        window.transient(self.root)

        body = scrolledtext.ScrolledText(window, wrap="word", padx=12, pady=12)
        body.pack(fill="both", expand=True)
        body.insert("end", HINTS_HELP_TEXT)
        # Make it read only, since it is there to be read and not edited.
        body.configure(state="disabled")

        ttk.Button(window, text="Close", command=window.destroy).pack(pady=8)

    def on_watch_clicked(self):
        # If watching is already on, this click turns it off.
        if self.watch_thread and self.watch_thread.is_alive():
            self.watch_stop.set()
            self.watch_thread = None
            self.watch_dir = None
            self.watch_button.config(text="Watch folder")
            self.watch_var.set("Watch mode is off.")
            return

        folder = filedialog.askdirectory(title="Pick a folder to watch")
        if not folder:
            return

        self.watch_dir = folder
        self.watch_stop = threading.Event()

        # Record the files already in the folder so turning on watch mode does not
        # kick off a big batch of old files by surprise. Watch mode reacts to files
        # you add after you turn it on, and it also skips anything that already has a
        # transcript next to it.
        self.prime_watch_folder(folder)

        self.watch_thread = threading.Thread(
            target=self.watch_loop, args=(folder, self.watch_stop), daemon=True
        )
        self.watch_thread.start()
        self.watch_button.config(text="Stop watching")
        self.watch_var.set(f"Watching: {folder}")

    # ----- queuing work --------------------------------------------------------

    def enqueue_job(self, path):
        """Put one file on the work queue. Safe to call from any thread."""
        key = str(Path(path).resolve())
        self.handled.add(key)
        job = {
            "path": str(path),
            "model": self.current_model,
            "hints": self.current_hints,
        }
        self.job_queue.put(job)
        self.ui_queue.put(("status", f"Queued: {Path(path).name}"))

    # ----- watch mode (runs on its own thread) ---------------------------------

    def prime_watch_folder(self, folder):
        try:
            for name in os.listdir(folder):
                full = Path(folder) / name
                if full.is_file() and full.suffix.lower() in AUDIO_VIDEO_EXTENSIONS:
                    self.handled.add(str(full.resolve()))
        except OSError:
            pass

    def watch_loop(self, folder, stop_event):
        poll_seconds = 2.0
        # A file has to hold the same size for this long before the app trusts that
        # the copy has finished. Grabbing a half written file produces garbage, so
        # this wait is worth it.
        stable_seconds = 3.0
        while not stop_event.is_set():
            try:
                self.scan_folder_once(folder, stable_seconds)
            except Exception as error:
                self.ui_queue.put(("status", f"Watch problem: {error}"))
            # Wait, but wake up early if watching is turned off.
            stop_event.wait(poll_seconds)

    def scan_folder_once(self, folder, stable_seconds):
        now = time.time()
        try:
            names = os.listdir(folder)
        except OSError:
            return

        for name in names:
            full = Path(folder) / name
            if not full.is_file():
                continue
            if full.suffix.lower() not in AUDIO_VIDEO_EXTENSIONS:
                continue

            key = str(full.resolve())
            if key in self.handled:
                continue
            if has_transcript(full):
                # Already transcribed at some point. Remember it so the app stops
                # checking it.
                self.handled.add(key)
                continue

            # Wait for the file to finish copying by watching its size settle.
            try:
                size = full.stat().st_size
            except OSError:
                continue

            previous = self.seen_sizes.get(key)
            if previous is None or previous[0] != size:
                # First time seeing this file, or the size changed since last time.
                # Record it and check again on the next pass.
                self.seen_sizes[key] = (size, now)
                continue

            first_seen_at = previous[1]
            if now - first_seen_at < stable_seconds:
                # The size has held steady, but not for long enough yet.
                continue
            if size == 0:
                # An empty file is not ready. Keep waiting.
                continue

            # Make sure the file is not still locked by whatever is copying it.
            try:
                with open(full, "rb"):
                    pass
            except OSError:
                # Still locked. Leave the size entry in place and try again later.
                continue

            # The file looks complete. Queue it and stop tracking its size.
            self.seen_sizes.pop(key, None)
            self.enqueue_job(full)

    # ----- the worker (runs on its own thread) ---------------------------------

    def worker_loop(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            path = job["path"]
            try:
                self.process_file(path, job["model"], job["hints"])
            except Exception as error:
                self.ui_queue.put(
                    ("status", f"Something went wrong with {Path(path).name}: {error}")
                )
                self.ui_queue.put(("progress", 0))
            finally:
                self.job_queue.task_done()

    def ensure_model(self, model_name):
        """Load the model if it is not loaded, or if the dropdown changed it.

        If the graphics card cannot start the model, this falls back to the
        processor so you still get a result. That usually means the CUDA 12
        libraries (cuBLAS and cuDNN) are not in place yet.
        """
        key = (model_name, self.device, self.compute_type)
        if self.model is not None and self.loaded_model_key == key:
            return

        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(
                model_name, device=self.device, compute_type=self.compute_type
            )
            self.loaded_model_key = key
        except Exception as gpu_error:
            if self.device == "cuda":
                self.device, self.compute_type = "cpu", "int8"
                self.ui_queue.put(
                    (
                        "device",
                        "Device: processor (cpu, int8). The graphics card did not "
                        "start, so the app fell back to the processor.",
                    )
                )
                self.ui_queue.put(
                    (
                        "status",
                        f"The graphics card did not start ({gpu_error}). Falling "
                        "back to the processor.",
                    )
                )
                self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
                self.loaded_model_key = (model_name, "cpu", "int8")
            else:
                raise

    def process_file(self, path, model_name, hints):
        source = Path(path)

        self.ui_queue.put(
            (
                "status",
                f"Loading the {model_name} model. The first time for a model this "
                "can take a while.",
            )
        )
        self.ui_queue.put(("progress", 0))
        self.ensure_model(model_name)

        # A header in the live view so you can tell files apart in watch mode.
        self.ui_queue.put(
            ("append", "\n" + "=" * 60 + f"\nFile: {source.name}\n" + "=" * 60 + "\n\n")
        )
        self.ui_queue.put(("status", f"Transcribing {source.name} ..."))

        # The settings here are chosen for accuracy on classroom and lecture audio.
        # language is forced to English so it does not guess the language. beam_size
        # of 5 searches a bit wider for the best wording. vad_filter turns on voice
        # activity detection, which basically means it skips long silences so it does
        # not invent text during quiet parts, and echoey rooms cause exactly that.
        # initial_prompt feeds your hints in up front so names and subject words are
        # spelled the way you typed them.
        segments_iter, info = self.model.transcribe(
            str(source),
            language="en",
            beam_size=5,
            vad_filter=True,
            initial_prompt=hints or None,
        )

        duration = info.duration or 0.0
        collected = []
        for segment in segments_iter:
            collected.append(segment)
            text = segment.text.strip()
            line = f"[{format_timestamp_txt(segment.start)}] {text}"
            self.ui_queue.put(("append", line + "\n"))
            if duration > 0:
                fraction = min(1.0, segment.end / duration)
                self.ui_queue.put(("progress", fraction * 100.0))

        if not collected:
            self.ui_queue.put(("append", "(No speech was found in this file.)\n"))

        txt_path = transcript_txt_path(source)
        srt_path = transcript_srt_path(source)
        self.write_txt(txt_path, collected)
        self.write_srt(srt_path, collected)

        self.last_output_dir = str(source.parent)
        self.ui_queue.put(("progress", 100))
        self.ui_queue.put(
            (
                "status",
                f"Done. Saved {txt_path.name} and {srt_path.name} next to the file.",
            )
        )
        self.ui_queue.put(("append", f"\nSaved:\n{txt_path}\n{srt_path}\n"))

    # ----- writing the output --------------------------------------------------

    def write_txt(self, txt_path, segments):
        # The text transcript puts the start time of each line in front of it in
        # [HH:MM:SS] form, so you can scrub back to that spot in the video later.
        with open(txt_path, "w", encoding="utf-8") as handle:
            for segment in segments:
                stamp = format_timestamp_txt(segment.start)
                handle.write(f"[{stamp}] {segment.text.strip()}\n")

    def write_srt(self, srt_path, segments):
        # The subtitle file holds the same lines from the same data, with a start
        # time and an end time, so a video player can show each line at the right
        # moment.
        with open(srt_path, "w", encoding="utf-8") as handle:
            for index, segment in enumerate(segments, start=1):
                start = format_timestamp_srt(segment.start)
                end = format_timestamp_srt(segment.end)
                handle.write(f"{index}\n")
                handle.write(f"{start} --> {end}\n")
                handle.write(f"{segment.text.strip()}\n\n")

    # ----- the main thread loop ------------------------------------------------

    def poll_ui_queue(self):
        # Keep the plain copies of the inputs current. This runs on the main thread,
        # so it is safe to read the widgets here, and the background threads read the
        # plain copies instead.
        try:
            self.current_model = self.model_var.get()
            self.current_hints = self.hints_text.get("1.0", "end").strip()
        except tk.TclError:
            pass

        # Apply every message the background threads have sent.
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "append":
                    self.output.insert("end", payload)
                    self.output.see("end")
                elif kind == "device":
                    self.device_var.set(payload)
        except queue.Empty:
            pass

        self.root.after(100, self.poll_ui_queue)


def main():
    # Make the CUDA libraries findable before anything tries to use the card.
    add_cuda_dll_directories()

    root = tk.Tk()
    root.geometry("840x680")
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
