# MeetingNotes

An automated Discord meeting-transcription pipeline (in progress). The
end goal: capture a Discord voice-channel recording, transcribe it locally,
and turn it into structured meeting notes automatically.

## Status

This is early-stage. What exists today are the two MVP building blocks,
built and tested independently — the Discord bot that would trigger them
automatically doesn't exist yet.

| Component | What it is | Status |
|---|---|---|
| [`transcription/`](transcription/) | Local Whisper transcription + OpenAI summarization pipeline | Working MVP — see testing below |
| [`MeetingNotes.Api/`](MeetingNotes.Api/) | ASP.NET Core Web API scaffold | Default template scaffold only, not yet wired to the pipeline |
| Discord bot integration | Watches a voice channel, records, triggers the pipeline | Not started |
| Speaker diarization | Distinguishing who said what | Solved for [Craig](https://craig.chat/) recordings via `transcribe_multitrack.py` (per-speaker tracks, no guessing needed); a single mixed-down file still relies on `transcribe.py`'s attribution heuristics |

## Repo structure

```
MeetingNotes/
├── transcription/              # Python: audio -> transcript -> structured notes
│   ├── transcribe.py           # local Whisper transcription, single file (offline)
│   ├── transcribe_multitrack.py # local Whisper transcription, Craig multi-track export (offline)
│   ├── summarize.py            # transcript -> Markdown notes (OpenAI API)
│   ├── pipeline.py             # chains transcribe.py + summarize.py
│   ├── input/                   # drop recordings here (gitignored); scripts auto-pick if path omitted
│   ├── output/                  # timestamped transcripts/notes land here (gitignored)
│   └── README.md               # full setup/usage/troubleshooting details
└── MeetingNotes.Api/           # ASP.NET Core Web API scaffold
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
Expect: structured Markdown notes (Summary / Discussion / Decisions Made /
Action Items / Milestones / Open Questions) printed to console and saved to
`output\<new timestamp>_<filename>_notes.md`.

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
[Craig](https://craig.chat/) multi-track Discord export:

```bash
python transcribe_multitrack.py path\to\craig-export-folder --model small --language en
```
Expect a merged, chronological, speaker-labeled transcript (ground-truth
speaker attribution, no guessing) saved to
`output\<timestamp>_<recording name>_multitrack.txt`. Takes roughly N times
as long as a single-file run (N = number of tracks) — see
`transcription/README.md` for real timing and why extra hallucination
filtering was needed here.

### `MeetingNotes.Api/` — the scaffold

This is currently the unmodified `dotnet new webapi -controllers` template,
not yet connected to the transcription pipeline.

```bash
cd MeetingNotes.Api
dotnet run
```

Expect console output ending in `Now listening on: http://localhost:5129`
(or similar). To confirm it's actually serving requests:

```bash
curl http://localhost:5129/openapi/v1.json
curl http://localhost:5129/WeatherForecast
```

Both should return `200` with JSON bodies. Note: this .NET version doesn't
ship a browsable Swagger UI page by default (only the raw OpenAPI JSON at
`/openapi/v1.json`) — `/swagger` will 404 unless Swashbuckle is added
separately.

## Next steps

1. Wire `MeetingNotes.Api` to the transcription pipeline (or decide it's
   not needed if the pipeline stays a standalone script/bot).
2. Build the Discord bot: watch a voice channel, record with Craig
   (multi-track), trigger `transcribe_multitrack.py` + `summarize.py`
   automatically.
3. Wire `transcribe_multitrack.py` into `pipeline.py` for one-command
   convenience (currently a manual two-step: run it, then feed the output
   into `summarize.py`).
