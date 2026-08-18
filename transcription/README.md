# MeetingNotes — Local Transcription (MVP)

This is the first building block of the MeetingNotes pipeline: turning a
recorded Discord audio file into a text transcript, fully offline, using
OpenAI's open-source [Whisper](https://github.com/openai/whisper) model.

No audio or text leaves your machine at this stage — no API calls. The next
step (not part of this script) will feed the resulting transcript into the
Anthropic Claude API to generate structured meeting notes.

## Setup

### 1. Install ffmpeg (required, not a Python package)

Whisper shells out to the `ffmpeg` binary to decode audio. Install it and
make sure it's on your PATH:

```bash
winget install Gyan.FFmpeg
```

or via Chocolatey:

```bash
choco install ffmpeg
```

or download a build manually from https://www.gyan.dev/ffmpeg/builds/,
extract it, and add its `bin\` folder to your PATH.

Verify it worked in a **new** terminal window:

```bash
ffmpeg -version
```

### 2. Install Python dependencies

From this `transcription/` folder:

```bash
pip install -r requirements.txt
```

This installs `openai-whisper` and its dependency `torch`. First run of the
script will also download the selected model's weights (one-time, cached
under `~/.cache/whisper`).

## Usage

```bash
python transcribe.py path\to\recording.flac
```

Use a different model size with `--model`:

```bash
python transcribe.py path\to\recording.flac --model small
```

Available models (fastest/least-accurate → slowest/most-accurate):
`tiny`, `base` (default), `small`, `medium`, `large`.

The transcript is:
- printed to the console, and
- saved to `output\<input_filename>.txt`

The script also prints how long transcription took, so you can gauge
performance on your hardware.

## CPU vs. GPU

Whisper runs fine on CPU-only machines — it'll just be slower.

**Correction from initial testing:** on Windows, `pip install torch` (which
`requirements.txt` pulls in via `openai-whisper`) installs the **CPU-only**
build by default, even on a machine with an NVIDIA GPU — confirmed on this
machine (GTX 1650 Ti present, but `torch.cuda.is_available()` returned
`False` until reinstalled with the line below). It does NOT auto-detect
CUDA the way it does on some other platforms.

To enable GPU acceleration on an NVIDIA card, install the CUDA build of
torch explicitly **before** installing `openai-whisper`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

(Check https://pytorch.org/get-started/locally/ for the CUDA version
matching your installed NVIDIA driver.)

Verify it worked:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Rough expectations for a **~30-minute** audio file with the `base` model:

| Hardware              | Approx. time     |
|------------------------|------------------|
| Modern CPU (multi-core) | ~5–15 minutes    |
| Entry-level NVIDIA GPU (CUDA torch installed) | ~1–3 minutes |

Larger models (`small`/`medium`/`large`) are noticeably slower — budget
2-5x longer on CPU as you go up in size.

### Confirmed on this machine

A short (~20s) synthetic test clip was transcribed successfully on CPU
(current setup, no CUDA torch installed): model load + transcription took
about 13 seconds total, with actual transcription at ~4 seconds. Extrapolating
CPU-only throughput, a real 30-minute recording should land in the
~5–15 minute range noted above.

## Error handling

The script checks for and reports, with a clear message instead of a raw
stack trace:
- missing/misspelled audio file path
- missing `ffmpeg` on PATH
- unsupported/corrupt audio format
- missing `openai-whisper` package
