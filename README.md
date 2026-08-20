# MeetingNotes

An automated Discord meeting-transcription pipeline (in progress). The
end goal: capture a Discord voice-channel recording, transcribe it locally,
and turn it into structured meeting notes automatically.

## Status

This is early-stage, and Python-only — an earlier ASP.NET Web API scaffold
that lived here was dropped as unused; the pipeline runs entirely as CLI
scripts.

| Component | What it is | Status |
|---|---|---|
| [`transcription/`](transcription/) | Local Whisper transcription + OpenAI summarization pipeline | Working MVP — see testing below |
| Discord bot integration | Paste a Craig recording link in Discord, get notes posted back automatically | Working — `craig_client.py` verified against a real live recording end-to-end (see `transcription/README.md` "Discord bot" section); a live `/meetingnotes` run through Discord itself is still worth doing as final confirmation |
| Speaker diarization | Distinguishing who said what | Solved for [Craig](https://craig.chat/) recordings via `transcribe_multitrack.py` (per-speaker tracks, no guessing needed); a single mixed-down file still relies on `transcribe.py`'s attribution heuristics |

## Repo structure

```
MeetingNotes/
└── transcription/              # Python: audio -> transcript -> structured notes
    ├── transcribe.py           # local Whisper transcription, single file (offline)
    ├── transcribe_multitrack.py # local Whisper transcription, Craig multi-track export (offline)
    ├── summarize.py            # transcript -> Markdown notes (OpenAI API)
    ├── pipeline.py             # auto-detects file vs. Craig folder, then chains transcription + summarize.py
    ├── craig_client.py         # fetches/cooks/downloads a recording from Craig's API
    ├── discord_bot.py          # Discord bot: paste a Craig link, get notes posted back
    ├── input/                   # drop recordings here (gitignored); scripts auto-pick if path omitted
    ├── output/                  # timestamped transcripts/notes land here (gitignored)
    └── README.md               # full setup/usage/troubleshooting details
```

## Testing

### `transcription/` — the working part

Full setup instructions (ffmpeg install, Python deps, API key, accuracy
tuning) are in [`transcription/README.md`](transcription/README.md). Once
set up:

```bash
cd transcription
```

**1. Transcription only** (fully offline, no API key needed — good first
smoke test):

```bash
python transcribe.py path\to\any_short_audio_file.wav
python transcribe.py                                   # or drop a file in input/ and omit the path
```
Expect: the transcript printed to console, saved to
`transcription/output/<timestamp>_<filename>.txt`, and a line reporting how
long it took. If you don't have a recording handy, any short
`.wav`/`.mp3`/`.flac` works for a smoke test — it doesn't need to be a real
meeting.

**2. Summarization only** (needs `OPENAI_API_KEY` set in
`transcription/.env` — see `transcription/README.md` §3):

```bash
python summarize.py output\<timestamp>_<filename>.txt
```
Expect: three passes printed as they run (extract → organize → verify —
this is deliberate, see `transcription/README.md` "Avoiding dropped
content"), then structured Markdown notes (Summary / Discussion / Decisions
Made / Action Items / Milestones / Open Questions) printed to console and
saved to `output\<new timestamp>_<filename>_notes.md`. Takes roughly 3x
longer/more API calls than a single-shot summary would — pass
`--single-pass` to trade completeness for speed on quick/low-stakes runs.

**3. Full pipeline** (both steps chained):

```bash
python pipeline.py path\to\recording.flac --model small --language en --initial-prompt "Docker, Git, MVP"
```
Expect both outputs saved, and a final summary block showing both output
paths and total elapsed time. Use `--skip-summary` to test just the
transcription half without needing an API key.

For real (non-smoke-test) accuracy, `--model small` or larger is strongly
recommended over the `base` default — see the "Improving accuracy" section
in `transcription/README.md` for why, with real before/after examples.

**4. Multi-track (Craig) transcription** — if you have a
[Craig](https://craig.chat/) multi-track Discord export, `pipeline.py`
auto-detects a folder vs. a single file and routes accordingly:

```bash
python pipeline.py path\to\craig-export-folder --model small --language en
python transcribe_multitrack.py path\to\craig-export-folder --model small   # transcription only
```
Expect a merged, chronological, speaker-labeled transcript (ground-truth
speaker attribution, no guessing) saved to
`output\<timestamp>_<recording name>_multitrack.txt`, then summarized the
same as the single-file path. Transcription takes roughly N times as long
as a single-file run (N = number of tracks) — see
`transcription/README.md` for real timing and why extra hallucination
filtering was needed here.

**5. Discord bot** — paste a Craig recording link into `/meetingnotes` and
get notes posted back in-channel automatically. Full setup (creating the
bot, permissions, `.env` config) is in `transcription/README.md`
"Discord bot". `craig_client.py` (the Craig API integration) has been
verified end-to-end against a real live recording — see that section for
what was tested and a real bug it caught (an unbounded error message
tripping Discord's own length limit).

## Next steps

1. Run a real `/meetingnotes` command through Discord itself as final
   confirmation — `craig_client.py` is verified via direct API calls, but
   hasn't yet been exercised through the bot's actual Discord interaction
   flow.
