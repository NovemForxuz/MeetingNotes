#!/usr/bin/env python3
"""
pipeline.py — chains transcription + summarize.py into one command.

Handles both input types automatically:
    - A single audio file -> transcribe.py (offline, single mixed track,
      Whisper has to guess who's talking)
    - A Craig multi-track export folder -> transcribe_multitrack.py
      (offline, per-speaker tracks, ground-truth speaker attribution)
...then summarize.py's three-pass pipeline (OpenAI API) turns the result
into structured meeting notes. transcribe.py, transcribe_multitrack.py, and
summarize.py all still work standalone; this just calls their reusable
functions in sequence so you don't have to run multiple commands by hand.

Usage:
    python pipeline.py path\to\recording.flac                    # single file
    python pipeline.py path\to\craig-export-folder                # Craig multi-track
    python pipeline.py                                            # auto-picks from ./input/
    python pipeline.py path\to\recording.flac --model small --language en --initial-prompt "Docker, Git, MVP"
    python pipeline.py path\to\craig-export-folder --name-map "novemforxuz=Heriz,shamgoh=Sham"
    python pipeline.py path\to\recording.flac --skip-summary
    python pipeline.py path\to\recording.flac --summary-model gpt-4o
    python pipeline.py path\to\recording.flac --participants "James, Heriz, Sham, Marcus, Aaron"
    python pipeline.py path\to\recording.flac --notes-file my_rough_notes.txt

Output (both saved under ./output/, sharing one timestamp so the pair is
easy to spot):
    - <timestamp>_<name>.txt / <timestamp>_<name>_multitrack.txt   (transcript)
    - <timestamp>_<name>_notes.md                                  (structured notes)
"""

import argparse
import sys
from pathlib import Path

import summarize
import transcribe
import transcribe_multitrack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file or Craig multi-track export locally, then "
        "summarize it into structured meeting notes via the OpenAI API."
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an audio file (e.g. a .flac recording) OR a Craig multi-track "
        "export folder. A bare name is also looked up inside ./input/. If omitted, "
        "auto-picks the single candidate (file or folder) found in ./input/ (errors "
        "if there's zero or more than one).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=transcribe.VALID_MODELS,
        help="Whisper model size for transcription (default: base). For a Craig folder, "
        "applied to every track.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force a language code (e.g. 'en') for transcription instead of auto-detecting.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Vocabulary hint text passed to Whisper to improve jargon recognition.",
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help='Craig multi-track only: comma-separated "discord_username=Real Name" pairs '
        'to relabel speakers, e.g. "novemforxuz=Heriz,shamgoh=Sham". Ignored for a '
        "single audio file.",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=summarize.DEFAULT_MODEL,
        help=f"OpenAI model for the summarization step (default: {summarize.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default=None,
        help='Comma-separated real participant names, e.g. "James, Heriz, Sham, Marcus, '
        'Aaron". Passed to the summarization step to correctly attribute action items '
        "when Whisper has mis-transcribed names. Usually unnecessary for a Craig "
        "multi-track input if you already used --name-map.",
    )
    parser.add_argument(
        "--notes-file",
        type=str,
        default=None,
        help="Path to a text file with your own rough meeting notes, used as a "
        "ground-truth cross-reference for the summarization step.",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Skip summarize.py's three-pass extract/organize/verify pipeline for one "
        "direct call instead. Faster/cheaper, but confirmed in testing to drop real "
        "content on long transcripts.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Only run transcription; skip the OpenAI summarization step (e.g. if you don't "
        "have an API key set up yet).",
    )
    return parser.parse_args()


def resolve_input(path_arg: str | None) -> tuple:
    """
    Figure out what to transcribe and which mode to use.

    Returns (path, mode) where mode is "single" (one audio file, ->
    transcribe.py) or "multitrack" (a Craig export folder, ->
    transcribe_multitrack.py).

    - Path given: if it resolves (as-is, or as a name inside ./input/) to a
      directory, that's multitrack; to a file, that's single. If it
      resolves to neither, defer to transcribe.resolve_audio_path()'s
      error message (keeps the "not found" wording consistent with
      transcribe.py's own CLI).
    - No path given: scan ./input/ for both audio files and subfolders
      that look like Craig exports (contain files matching Craig's
      "<track#>-<username>.<ext>" naming). Errors if the combined
      candidate count isn't exactly one.
    """
    if path_arg is not None:
        given = Path(path_arg).expanduser()
        candidate = given if given.exists() else (transcribe.INPUT_DIR / path_arg)
        if candidate.exists():
            return (candidate.resolve(), "multitrack") if candidate.is_dir() else (candidate.resolve(), "single")
        # Neither exists — let transcribe.resolve_audio_path() produce its usual
        # "not found" error for a consistent message.
        return transcribe.resolve_audio_path(path_arg), "single"

    transcribe.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_candidates = sorted(
        p for p in transcribe.INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in transcribe.SUPPORTED_EXTENSIONS
    )
    dir_candidates = sorted(
        p for p in transcribe.INPUT_DIR.iterdir()
        if p.is_dir() and transcribe_multitrack.find_tracks(p)
    )
    candidates = [(p, "single") for p in file_candidates] + [(p, "multitrack") for p in dir_candidates]

    if not candidates:
        print(
            f"ERROR: No audio file or Craig export folder found in {transcribe.INPUT_DIR}.\n"
            f"Either pass a path, or drop one into that folder.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(candidates) > 1:
        names = "\n".join(f"  - {p.name} ({mode})" for p, mode in candidates)
        print(
            f"ERROR: Multiple candidates found in {transcribe.INPUT_DIR}, and no path was "
            f"given to disambiguate:\n{names}\nPass one explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    path, mode = candidates[0]
    return path.resolve(), mode


def main() -> None:
    args = parse_args()
    input_path, mode = resolve_input(args.input_path)
    timestamp = transcribe.make_timestamp()

    if mode == "multitrack":
        print("=" * 60)
        print("STEP 1/2: MULTI-TRACK TRANSCRIPTION (Craig, local, offline)")
        print("=" * 60)
        name_map = transcribe_multitrack.parse_name_map(args.name_map)
        transcription_result = transcribe_multitrack.transcribe_multitrack(
            input_path,
            model_name=args.model,
            language=args.language,
            initial_prompt=args.initial_prompt,
            name_map=name_map,
            timestamp=timestamp,
        )
        base_name = transcribe_multitrack.get_recording_name(input_path)
    else:
        print("=" * 60)
        print("STEP 1/2: TRANSCRIPTION (local, offline)")
        print("=" * 60)
        transcription_result = transcribe.transcribe_audio(
            input_path,
            model_name=args.model,
            language=args.language,
            initial_prompt=args.initial_prompt,
            timestamp=timestamp,
        )
        base_name = input_path.stem

    if args.skip_summary:
        print("\n--skip-summary set: stopping after transcription.")
        return

    print("\n" + "=" * 60)
    print("STEP 2/2: SUMMARIZATION (OpenAI API, requires network)")
    print("=" * 60)
    notes_text = summarize.check_notes_file(args.notes_file)
    summary_result = summarize.summarize_and_save(
        transcription_result["transcript"],
        base_name,
        model=args.summary_model,
        participants=args.participants,
        notes_text=notes_text,
        single_pass=args.single_pass,
        timestamp=timestamp,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Transcript:    {transcription_result['output_path']}")
    print(f"Meeting notes: {summary_result['output_path']}")
    total_elapsed = transcription_result["elapsed_seconds"] + summary_result["elapsed_seconds"]
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed / 60:.1f} minutes).")


if __name__ == "__main__":
    main()
