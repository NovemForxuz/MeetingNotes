#!/usr/bin/env python3
"""
pipeline.py — chains transcribe.py + summarize.py into one command.

Discord audio file -> local Whisper transcript -> structured meeting notes
(via the OpenAI API), end to end. transcribe.py and summarize.py both still
work standalone; this just calls their reusable functions in sequence so you
don't have to run two commands by hand.

Usage:
    python pipeline.py path\to\recording.flac
    python pipeline.py                              # auto-picks the one file in ./input/
    python pipeline.py path\to\recording.flac --model small --language en --initial-prompt "Docker, Git, MVP"
    python pipeline.py path\to\recording.flac --skip-summary
    python pipeline.py path\to\recording.flac --summary-model gpt-4o
    python pipeline.py path\to\recording.flac --participants "James, Heriz, Sham, Marcus, Aaron"
    python pipeline.py path\to\recording.flac --notes-file my_rough_notes.txt

Output (both saved under ./output/, sharing one timestamp so the pair is
easy to spot):
    - <timestamp>_<input filename stem>.txt        (transcript, from transcribe.py)
    - <timestamp>_<input filename stem>_notes.md   (structured notes, from summarize.py)
"""

import argparse
import sys
from pathlib import Path

import summarize
import transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file locally, then summarize it into structured "
        "meeting notes via the OpenAI API."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the audio file to transcribe (e.g. a .flac recording from Discord). "
        "A bare filename is also looked up inside ./input/. If omitted, auto-picks the "
        "single audio file in ./input/ (errors if there's zero or more than one).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=transcribe.VALID_MODELS,
        help="Whisper model size for transcription (default: base).",
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
        "when Whisper has mis-transcribed names.",
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


def main() -> None:
    args = parse_args()
    audio_path = transcribe.resolve_audio_path(args.audio_path)
    timestamp = transcribe.make_timestamp()

    print("=" * 60)
    print("STEP 1/2: TRANSCRIPTION (local, offline)")
    print("=" * 60)
    transcription_result = transcribe.transcribe_audio(
        audio_path,
        model_name=args.model,
        language=args.language,
        initial_prompt=args.initial_prompt,
        timestamp=timestamp,
    )

    if args.skip_summary:
        print("\n--skip-summary set: stopping after transcription.")
        return

    print("\n" + "=" * 60)
    print("STEP 2/2: SUMMARIZATION (OpenAI API, requires network)")
    print("=" * 60)
    notes_text = summarize.check_notes_file(args.notes_file)
    summary_result = summarize.summarize_and_save(
        transcription_result["transcript"],
        audio_path.stem,
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
