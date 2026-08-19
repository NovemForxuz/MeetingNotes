# MeetingNotes — Local Transcription + Summarization (MVP)

This is the MeetingNotes MVP pipeline: turning a recorded Discord audio file
into structured meeting notes, in two chained steps.

| Script | Does what | Offline? |
|---|---|---|
| `transcribe.py` | One audio file → text transcript, via local [Whisper](https://github.com/openai/whisper) | Yes — fully offline, no API calls |
| `transcribe_multitrack.py` | A [Craig](https://craig.chat/) multi-track Discord export → one merged, speaker-labeled transcript | Yes — fully offline, no API calls |
| `summarize.py` | Transcript → structured Markdown notes, via the OpenAI API | No — needs network + an API key |
| `pipeline.py` | Runs `transcribe.py` + `summarize.py` in sequence | No (unless `--skip-summary`) |

Each script also works standalone with its own CLI — `pipeline.py` just
imports their reusable functions (`transcribe_audio()`, `summarize_and_save()`)
and calls them one after another; it doesn't duplicate any logic.
`transcribe_multitrack.py` isn't wired into `pipeline.py` yet — run it, then
feed its output transcript into `summarize.py` manually (see below).

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

This installs `openai-whisper`/`torch` (transcription) and `openai`/
`python-dotenv` (summarization). First run of `transcribe.py` will also
download the selected Whisper model's weights (one-time, cached under
`~/.cache/whisper`).

### 3. Set up your OpenAI API key (only needed for summarize.py / pipeline.py)

Copy `.env.example` to `.env` in this folder and fill in a real key:

```bash
copy .env.example .env
```

```
OPENAI_API_KEY=your-key-here
```

Get a key at https://platform.openai.com/api-keys. `.env` is gitignored —
never commit real keys. `transcribe.py` alone doesn't need this; only
`summarize.py` and `pipeline.py` (unless run with `--skip-summary`) do.

> If you already had a `.env` file from an earlier version of this project
> with `ANTHROPIC_API_KEY`, rename that variable to `OPENAI_API_KEY` (with a
> real OpenAI key as the value) — summarization now calls OpenAI, not
> Anthropic.

## Usage

### Dropping files in `input/`

All three scripts accept an explicit path, but you can also just drop a
recording into `transcription/input/` and omit the path argument — it'll
auto-pick the one audio file found there:

```bash
python transcribe.py                    # auto-picks the single file in input/
python transcribe.py recording.flac     # looked up inside input/ if not found as-is
python pipeline.py                      # same auto-pick behavior
```

If `input/` has zero or more than one audio file, you'll get a clear error
telling you what's there instead of it guessing. `input/` is gitignored,
same as `output/` — it's meant to hold real recordings locally, not get
committed.

### Output filenames

Every run prefixes its output filename with a timestamp
(`YYYYMMDD_HHMMSS_`), so repeated runs never silently overwrite each
other. `pipeline.py` uses one shared timestamp for both the transcript and
the notes file, so the pair is easy to spot together, e.g.:

```
output/20260818_225216_recording.txt
output/20260818_225216_recording_notes.md
```

### Full pipeline (transcribe + summarize)

```bash
python pipeline.py path\to\recording.flac
python pipeline.py path\to\recording.flac --model small --language en --initial-prompt "Docker, Git, MVP"
python pipeline.py path\to\recording.flac --skip-summary   # transcription only, no API key needed
```

Saves both `output\<timestamp>_<name>.txt` (transcript) and
`output\<timestamp>_<name>_notes.md` (structured notes).

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
- saved to `output\<timestamp>_<input_filename>.txt`

The script also prints:
- which device it used (GPU/CPU — auto-detected)
- the language Whisper detected
- how long transcription took, so you can gauge performance on your hardware

### Multi-track transcription (Craig)

If you record with the [Craig](https://craig.chat/) Discord bot, it captures
each speaker to a **separate audio track** instead of one mixed-down file.
That sidesteps `transcribe.py`'s single biggest limitation: on a mixed file,
Whisper has to *guess* who's talking from context; with per-speaker tracks,
we know who said what for free.

In Discord, download the recording as `flac` (or `wav`/`mp3`/etc — any
format `transcribe.py` supports) and extract the resulting `.zip`. You'll
get a folder with `info.txt`, one audio file per speaker named
`<track#>-<discord username>.<ext>`, and (ignorable for our purposes) a
`raw.dat`. Point `transcribe_multitrack.py` at that folder:

```bash
python transcribe_multitrack.py path\to\craig-export-folder
python transcribe_multitrack.py path\to\craig-export-folder --model small --language en
python transcribe_multitrack.py path\to\craig-export-folder --name-map "novemforxuz=Heriz,shamgoh=Sham"
```

It transcribes each track independently (keeping per-segment timestamps,
not just flattened text), drops segments that look like silence/
hallucination on that track (see "Why the extra filtering" below), then
merges every kept segment from every track into one chronological
transcript:

```
[03:00] Heriz: the state escalation or like based on my three conditions.
[03:34] Sham: ...
```

Use `--name-map` to relabel Discord usernames to real names in the output
(usernames are used as-is if you skip it). Saved to
`output\<timestamp>_<recording name>_multitrack.txt` — feed that into
`summarize.py` exactly like a normal transcript. Since speakers are already
labeled, `--participants` is probably no longer needed there, though
`--notes-file` is still worth using.

**Verify the username→real-name mapping once, then reuse it.** Don't guess
at `--name-map` from usernames alone — confirm each mapping against actual
transcript content first (e.g. `grep` the merged transcript for something
you know that person said). Confirmed in testing on a real recording:
relabeling *before* summarization (so the transcript says "Heriz:" instead
of "novemforxuz:") measurably improved attribution accuracy over passing
`--participants` alone — the summarization model no longer has to resolve
third-person references (e.g. one speaker calling another "Harris" mid-
sentence) against a raw username at all. For a recurring meeting with the
same people, verify the mapping once and reuse the same `--name-map` on
every future run — no need to re-verify each time.

**Heads up on runtime:** this transcribes N tracks independently, so it
takes roughly N times as long as a single-file `transcribe.py` run at the
same model size — a 5-person, ~30-minute meeting at `--model small` took
this project's real test run on the order of tens of minutes on CPU.
Consider running it in the background.

**Why the extra filtering:** every track spans the *entire* meeting, most
of which is silence for any one speaker. Whisper's `no_speech_prob` alone
didn't catch everything in testing — background noise/breathing that isn't
silence but also isn't real speech produced fabricated text with reported
low `no_speech_prob` but very low confidence (`avg_logprob` measured around
-4.8 on real garbage output here, vs. -0.6 to -0.9 for genuine speech).
`transcribe_multitrack.py` filters on both.

### Summarization only

Given an existing transcript (e.g. from a previous `transcribe.py` run):

```bash
python summarize.py output\recording.txt
python summarize.py output\recording.txt --model gpt-4o
python summarize.py output\recording.txt --participants "James, Heriz, Sham, Marcus, Aaron"
python summarize.py output\recording.txt --notes-file my_rough_notes.txt
```

Produces structured Markdown notes with sections: Summary, Discussion
(grouped by speaker), Decisions Made, Action Items (with owner + due date
when the transcript states them), Milestones (dates/deadlines mentioned),
and Open Questions. Saved to `output\<timestamp>_<name>_notes.md`, and also
printed to console along with how long the API call took.

**Use `--participants`** with the real names of meeting attendees when you
have them — Whisper regularly mis-hears names (confirmed in testing: "Heriz"
came through as "Harris"/"Harry's" throughout a real transcript), and this
lets the summarization model map garbled names back to the right person and
attribute action items correctly instead of dropping them or marking them
"Unassigned". It can only fix names Whisper *did* transcribe, just wrong —
if Whisper missed a name entirely, try adding it to `transcribe.py`'s
`--initial-prompt` instead, so Whisper has a better shot at catching it in
the first place.

**Use `--notes-file`** if you also took your own rough notes during the
meeting — this is the single biggest accuracy lever available. A raw
Whisper transcript is often genuinely ambiguous even to a careful reader
(confusing overlapping dialogue, names Whisper never caught at all, garbled
deadline negotiations), and no amount of prompt tuning fully closes that
gap from the transcript alone. Your notes give the model a ground-truth
cross-reference: confirmed in testing that adding a short rough-notes file
correctly surfaced an action item owner (Sham) who has zero recognizable
mention anywhere in the Whisper transcript, and matched several other
action items that a transcript-only run had missed or gotten vague. The
notes don't need to be tidy — the shorthand you'd normally jot for
yourself is enough; the model treats them as authoritative over the
transcript when the two conflict, and uses the transcript to fill in
detail your notes only summarized.

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
- missing `openai` package
- missing/invalid `OPENAI_API_KEY`
- OpenAI API errors: auth failure, rate limit, network/connection issue,
  model not found (e.g. if `gpt-4o-mini` is renamed/deprecated by the time
  you read this — pass `--model` with a current one), other API-side errors

`pipeline.py` surfaces whichever of the above happens, from either step.
