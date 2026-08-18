#!/usr/bin/env python3
"""
summarize.py — turn a meeting transcript into structured notes using the
Anthropic Claude API.

Second step of the MeetingNotes MVP pipeline: transcript (from transcribe.py)
-> structured meeting notes (Markdown). Requires network access and an
Anthropic API key — unlike transcribe.py, this step is NOT offline.

Usage:
    python summarize.py path\to\transcript.txt
    python summarize.py path\to\transcript.txt --model claude-sonnet-5

Requires ANTHROPIC_API_KEY to be set, either as an environment variable or
in a .env file in this folder. See README.md.

Output:
    - Structured meeting notes (Markdown) printed to console
    - Saved to ./output/<transcript filename stem>_notes.md
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; ANTHROPIC_API_KEY can still be a real env var.

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are an assistant that turns raw meeting transcripts into clear, \
structured meeting notes. The transcript comes from local speech-to-text and may \
contain transcription errors, especially on technical jargon and names — use \
surrounding context to infer the likely intended word where a mis-transcription \
is obvious, but do not invent details the transcript doesn't support.

Structure your response as Markdown with these sections (omit a section only \
if genuinely nothing in the transcript fits it):

## Summary
2-3 sentence overview of what the meeting covered.

## Topics Discussed
Bullet list of topics, each with a brief description.

## Decisions Made
Bullet list of concrete decisions reached.

## Action Items
Bullet list formatted as "- [Owner]: Task (due: date if mentioned)". Use \
"Unassigned" when no owner is stated in the transcript.

## Open Questions
Bullet list of unresolved items explicitly left for later.

Output only the Markdown notes — no preamble, no commentary about the transcript."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured meeting notes from a transcript using the Anthropic Claude API."
    )
    parser.add_argument(
        "transcript_path",
        type=str,
        help="Path to a transcript .txt file (e.g. output from transcribe.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def check_transcript_file(transcript_path: Path) -> str:
    if not transcript_path.exists():
        print(f"ERROR: Transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)
    if not transcript_path.is_file():
        print(f"ERROR: Path is not a file: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"ERROR: Transcript file is empty: {transcript_path}", file=sys.stderr)
        sys.exit(1)
    return text


def import_anthropic():
    try:
        import anthropic
    except ImportError:
        print(
            "ERROR: The 'anthropic' package is not installed.\n"
            "Install dependencies with:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    return anthropic


def get_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Set it as an environment variable, or create a .env file in this\n"
            "folder containing:\n"
            "    ANTHROPIC_API_KEY=your-key-here\n\n"
            "Get a key at https://console.anthropic.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def summarize_transcript(transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Call the Claude API to turn a transcript into structured meeting notes.

    Returns the Markdown notes text. Exits the process on unrecoverable
    errors (missing package, missing/bad API key, API failure) — consistent
    with transcribe.py's error-handling style so both scripts behave the
    same whether run standalone or chained from pipeline.py.
    """
    anthropic = import_anthropic()
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Transcript:\n\n{transcript_text}"}],
        )
    except anthropic.AuthenticationError as exc:
        print(f"ERROR: Anthropic API authentication failed (check your API key): {exc}", file=sys.stderr)
        sys.exit(1)
    except anthropic.RateLimitError as exc:
        print(f"ERROR: Anthropic API rate limit hit: {exc}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError as exc:
        print(f"ERROR: Could not reach the Anthropic API (network issue?): {exc}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as exc:
        print(f"ERROR: Anthropic API returned an error (status {exc.status_code}): {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - last-resort catch-all so we fail with a readable message
        print(f"ERROR: Summarization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    return response.content[0].text.strip()


def summarize_and_save(
    transcript_text: str,
    base_name: str,
    model: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
) -> dict:
    """
    Summarize transcript_text and save the result to
    <output_dir>/<base_name>_notes.md. Prints the notes and timing.

    Returns a dict with "notes", "output_path", "elapsed_seconds".
    """
    output_dir = output_dir or (Path(__file__).parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_notes.md"

    print(f"Generating structured meeting notes with Claude ({model})...")
    start_time = time.perf_counter()
    notes = summarize_transcript(transcript_text, model=model)
    elapsed = time.perf_counter() - start_time

    output_path.write_text(notes, encoding="utf-8")

    print("\n" + "=" * 60)
    print("MEETING NOTES")
    print("=" * 60)
    print(notes)
    print("=" * 60)
    print(f"\nSaved meeting notes to: {output_path}")
    print(f"Summarization took {elapsed:.1f} seconds.")

    return {"notes": notes, "output_path": output_path, "elapsed_seconds": elapsed}


def main() -> None:
    args = parse_args()
    transcript_path = Path(args.transcript_path).expanduser().resolve()
    transcript_text = check_transcript_file(transcript_path)
    summarize_and_save(transcript_text, transcript_path.stem, model=args.model)


if __name__ == "__main__":
    main()
