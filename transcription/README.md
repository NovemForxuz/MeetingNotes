# MeetingNotes — Local Transcription + Summarization (MVP)

This is the MeetingNotes MVP pipeline: turning a recorded Discord audio file
into structured meeting notes, in two chained steps.

| Script | Does what | Offline? |
|---|---|---|
| `transcribe.py` | Audio → text transcript, via local [Whisper](https://github.com/openai/whisper) | Yes — fully offline, no API calls |
| `summarize.py` | Transcript → structured Markdown notes, via the Anthropic Claude API | No — needs network + an API key |
| `pipeline.py` | Runs both of the above in sequence | No (unless `--skip-summary`) |

Each script also works standalone with its own CLI — `pipeline.py` just
imports their reusable functions (`transcribe_audio()`, `summarize_and_save()`)
and calls them one after another; it doesn't duplicate any logic.

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

This installs `openai-whisper`/`torch` (transcription) and `anthropic`/
`python-dotenv` (summarization). First run of `transcribe.py` will also
download the selected Whisper model's weights (one-time, cached under
`~/.cache/whisper`).

### 3. Set up your Anthropic API key (only needed for summarize.py / pipeline.py)

Copy `.env.example` to `.env` in this folder and fill in a real key:

```bash
copy .env.example .env
```

```
ANTHROPIC_API_KEY=your-key-here
```

Get a key at https://console.anthropic.com/settings/keys. `.env` is
gitignored — never commit real keys. `transcribe.py` alone doesn't need
this; only `summarize.py` and `pipeline.py` (unless run with
`--skip-summary`) do.

## Usage

### Full pipeline (transcribe + summarize)

```bash
python pipeline.py path\to\recording.flac
python pipeline.py path\to\recording.flac --model small --language en --initial-prompt "Docker, Git, MVP"
python pipeline.py path\to\recording.flac --skip-summary   # transcription only, no API key needed
```

Saves both `output\<name>.txt` (transcript) and `output\<name>_notes.md`
(structured notes).

### Transcription only

```bash
python transcribe.py path\to\recording.flac
```

Use a different model size with `--model`:

```bash
python transcribe.py path\to\recording.flac --model small
```

Available models (fastest/least-accurate → slowest/most-accurate):
`tiny`, `base` (default), `small`, `medium`, `large`.

Force a language (skip auto-detection) and/or prime Whisper with expected
vocabulary — both help a lot with jargon-heavy meetings (see
"Improving accuracy" below):

```bash
python transcribe.py path\to\recording.flac --model small --language en --initial-prompt "Docker, CI/CD, ASP.NET Core, C#, .NET, Git, MVP"
```

The transcript is:
- printed to the console, and
- saved to `output\<input_filename>.txt`

The script also prints:
- which device it used (GPU/CPU — auto-detected)
- the language Whisper detected
- how long transcription took, so you can gauge performance on your hardware

### Summarization only

Given an existing transcript (e.g. from a previous `transcribe.py` run):

```bash
python summarize.py output\recording.txt
python summarize.py output\recording.txt --model claude-sonnet-5
```

Produces structured Markdown notes with sections: Summary, Topics Discussed,
Decisions Made, Action Items (with owner + due date when the transcript
states them), and Open Questions. Saved to `output\<name>_notes.md`, and
also printed to console along with how long the API call took.

## Improving accuracy

The `base` model badly mangles technical/domain jargon — confirmed on a real
test recording, where it turned "Docker" into "darker", "containers" into
"condoms", and "ASP.NET Core" into "PSP dot net core". This is a known
`base`-model weakness, not a bug in the script. Two things that measurably
helped in testing against that same file:

1. **Use `--model small` (or larger).** This did most of the work — jargon
   that `base` completely mangled came through legibly with `small`
   ("Docker", "Git", "ASP.NET Core", "Discord integration", "webhooks or
   polling" were all correctly transcribed).
2. **Use `--initial-prompt` with your recurring vocabulary** (product names,
   acronyms, tech stack terms). This primes Whisper's decoder toward the
   right words and sharpened jargon recognition further on top of the
   model-size improvement.

Also worth checking: **the language Whisper detected**, printed after each
run. On the same test file, Whisper's auto-detect locked onto `ms` (Malay)
even though the audio was English — auto-detection only samples the first
~30 seconds and can drift, especially with accented speech. Forcing
`--language en` didn't fix `base`'s jargon problem by itself, but combined
with `small` + `--initial-prompt` it gave the best result of everything
tested, including correctly picking up "Discord.NET" as a specific term.

None of this fixes multi-speaker separation — Whisper has no concept of
"who's talking." If multiple people overlap, expect blending/garbling
regardless of model size; that would need a separate diarization step
(e.g. `pyannote.audio`), out of scope for this MVP.

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

Measured on CPU (no CUDA torch installed) against a real ~105-second test
recording:

| Config                                              | Time    |
|------------------------------------------------------|---------|
| `base`, auto-detect language                          | ~18–20s |
| `small` + `--language en` + `--initial-prompt`        | ~32s    |

Roughly 2x slower going from `base` to `small`, consistent with the guidance
above. Extrapolating to a real 30-minute recording (~17x longer than this
test clip), expect `small` to land around **9–10 minutes** on this machine's
CPU — comfortably within the ~5–15 minute range estimated for `base`, since
the accuracy gain from `small` is worth the modest extra time.

## Error handling

Both scripts check for and report, with a clear message instead of a raw
stack trace:

**transcribe.py:**
- missing/misspelled audio file path
- missing `ffmpeg` on PATH
- unsupported/corrupt audio format
- missing `openai-whisper` package

**summarize.py:**
- missing/misspelled transcript file path, or an empty transcript
- missing `anthropic` package
- missing/invalid `ANTHROPIC_API_KEY`
- Anthropic API errors: auth failure, rate limit, network/connection issue,
  other API-side errors

`pipeline.py` surfaces whichever of the above happens, from either step.
