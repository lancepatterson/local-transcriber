# Local Transcriber

This is a small desktop app that turns audio and video into text transcripts on your own computer. It uses faster-whisper, which basically means a faster and lighter version of OpenAI's Whisper speech to text model. faster-whisper reads audio and video on its own, so you do not need a separate ffmpeg install.

It runs on your NVIDIA card when one is available, and it falls back to the processor when one is not. It detects the card for you, so there is nothing to set.

## A note on the internet

The app does not use any paid service, and it does not send your audio anywhere. The one exception is the model itself. The first time you use a given model, the app downloads it one time from the internet and saves it on your computer. After that, everything runs offline. So the very first run of a model needs the internet for that one download, and every run after that does not.

## What you need

You need Windows, Python, and for the fast path an NVIDIA card. The processor path works without a card, just slower.

Run all of the commands below in PowerShell. Each one is written out in full so you can copy it as is.

## Step 1. Install faster-whisper

This installs the latest faster-whisper and pulls in CTranslate2, which is the engine it runs on.

```
python -m pip install -U faster-whisper
```

That is all you need for the processor path. If you only ever want the processor, you can stop here and skip to Step 3.

## Step 2. Set up NVIDIA GPU support on Windows

Recent NVIDIA cards need current libraries. The 50 series in particular uses NVIDIA's newest architecture, called Blackwell, which the older libraries do not fully support. There are three pieces: a recent driver, the latest faster-whisper (done in Step 1), and the CUDA 12 libraries that faster-whisper needs. CUDA is NVIDIA's toolkit for running general work on the card, and the two libraries the app needs from it are cuBLAS (fast math on the card) and cuDNN version 9 (the neural network pieces).

### 2a. Install a recent NVIDIA driver

Download and install the latest Game Ready or Studio driver for your NVIDIA card.

```
https://www.nvidia.com/Download/index.aspx
```

Then confirm the driver is in place and the card is visible. This prints the driver version and the card name.

```
nvidia-smi
```

### 2b. Install the CUDA 12 libraries (cuBLAS and cuDNN 9)

These are official NVIDIA packages from pip. You do not need the full CUDA Toolkit, just these two runtime libraries. The cuDNN one is pinned to version 9 because that is the version current faster-whisper expects.

```
python -m pip install -U nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9.0,<10.0"
```

### 2c. Make sure Python can find them

The app does this part for you. When it starts, it finds the folders where those two libraries installed and adds them to the search path, so Python can load them without you editing PATH by hand. The reason this is needed is that pip puts the library files inside site-packages, and Windows does not look there for these files on its own.

If for some reason the card still does not start, you can add the folders to PATH yourself for the current PowerShell session with the command below, then run the app from that same window.

```
$env:PATH = (python -c "import os,nvidia.cublas;print(os.path.join(os.path.dirname(nvidia.cublas.__file__),'bin'))") + ";" + (python -c "import os,nvidia.cudnn;print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),'bin'))") + ";" + $env:PATH
```

### A word on float16 and int8

On the card the app uses float16, which is just a compact number format. It does not use int8 on the card on purpose. int8 currently crashes on Blackwell cards with a cuBLAS error, so the app stays on float16 for the card no matter what. On the processor the app uses int8, which is fine there and keeps the processor quick enough to be usable.

## Step 3. Run the app

From the folder that holds the file, start the app.

```
python local_transcriber.py
```

The window opens, and the Device line near the bottom tells you whether it is on the graphics card or the processor.

## How to use it

Select file picks one audio or video file. Press Transcribe to start it.

Model is the dropdown. It defaults to large-v2, because large-v2 held up better than the newer large-v3 on classroom and lecture audio in the testing this app was built around, and lecture audio is what you are feeding it. You can switch to large-v3, distil-large-v3, or small.en when you want to try them.

Hints is optional. Whatever you type goes in as the initial_prompt, which basically means you can give it the teacher's name and the subject vocabulary up front so it spells those correctly instead of guessing.

Show folder opens the folder where the last transcript was saved.

Watch folder is the main one. You click it and pick a folder, and from then on, while the app is open, any new audio or video file you drop into that folder gets transcribed on its own with no clicks from you. The manual Select file mode still works at the same time. Click the button again to stop watching.

Watch mode behaves carefully. It only reacts to files you add after you turn it on, and it skips any file that already has a transcript next to it, so it does not redo the whole folder. Before it starts a file, it waits until the file size stops changing, because otherwise it would grab a half written file and produce garbage. It only watches audio and video file types, so the text files and subtitle files it writes are never treated as new input.

The transcription runs on a background thread, which basically means the window stays responsive while a long file is processing.

## What it saves

For each file, the app saves two files next to the original.

The text transcript is named like the source with `_transcript` added, for example `lecture1_transcript.txt`. Each line has the start time in front of it in `[HH:MM:SS]` form, so you can scrub back to that spot in the video later to check a quote.

The subtitle file uses the same name with the `.srt` ending, for example `lecture1_transcript.srt`. It holds the same lines from the same data, so you can load it into a video player and see each line at the right time.

## How to confirm it works

1. The window opens. Run `python local_transcriber.py` and check that the window appears and the Device line reads `graphics card (cuda, float16)`.

2. You can transcribe one file by hand. Press Select file, pick a short audio or video file, press Transcribe, and watch the lines appear in the scrolling area. When it finishes, press Show folder and check that the `_transcript.txt` and `_transcript.srt` files are there, with `[HH:MM:SS]` times in the text file.

3. Watch mode works. Press Watch folder and pick an empty test folder. Copy an audio or video file into that folder. Within a few seconds the app starts on its own and saves a timestamped transcript next to the file.

4. It is using the card and not the processor. The fastest check is the Device line in the app. To be sure, start a transcription, and while it runs, open a second PowerShell window and run `nvidia-smi`. You should see a python process listed and the GPU-Util number climb. You can also check that the card is visible to the engine at any time with the command below, which prints the number of cards it can see.

```
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

If that last command prints `0`, the engine cannot see the card, and the app will run on the processor. Go back through Step 2, confirm `nvidia-smi` shows the card, and confirm the libraries from Step 2b installed without errors.

## If the card does not start

If the card is visible but the model fails to load on it, the app does not stop. It falls back to the processor so you still get a result, and it says so on the Device line and in the status message. That almost always means the libraries from Step 2b are missing or were not found, so check those first.
