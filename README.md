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
| Speaker diarization | Distinguishing who said what | Not started (Whisper alone doesn't do this) |

## Repo structure

```
MeetingNotes/
├── transcription/       # Python: audio -> transcript -> structured notes
│   ├── transcribe.py    # local Whisper transcription (offline)
│   ├── summarize.py     # transcript -> Markdown notes (OpenAI API)
│   ├── pipeline.py      # chains the two above
│   ├── input/            # drop recordings here (gitignored); scripts auto-pick if path omitted
│   ├── output/           # timestamped transcripts/notes land here (gitignored)
│   └── README.md        # full setup/usage/troubleshooting details
└── MeetingNotes.Api/    # ASP.NET Core Web API scaffold
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
2. Build the Discord bot: watch a voice channel, record, trigger
   `pipeline.py` automatically.
3. Add speaker diarization so multi-person meetings don't blend speakers
   together in the transcript.
