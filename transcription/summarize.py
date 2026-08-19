#!/usr/bin/env python3
"""
summarize.py — turn a meeting transcript into structured notes using the
OpenAI API.

Second step of the MeetingNotes MVP pipeline: transcript (from transcribe.py)
-> structured meeting notes (Markdown). Requires network access and an
OpenAI API key — unlike transcribe.py, this step is NOT offline.

Usage:
    python summarize.py path\to\transcript.txt
    python summarize.py path\to\transcript.txt --model gpt-4o
    python summarize.py path\to\transcript.txt --participants "James, Heriz, Sham, Marcus, Aaron"
    python summarize.py path\to\transcript.txt --notes-file my_rough_notes.txt

Requires OPENAI_API_KEY to be set, either as an environment variable or in a
.env file in this folder. See README.md.

Tip: local Whisper transcription regularly mis-hears names (e.g. "Heriz"
came through as "Harris"/"Harry's" in testing). Pass --participants with
the real names so the model can map garbled transcript names back to the
right person instead of dropping them or marking items "Unassigned".

Tip: if you also have your own rough notes from the meeting, pass them with
--notes-file. The transcript alone is often genuinely ambiguous (confusing
back-and-forth, names Whisper never caught at all) — your notes give the
model a ground-truth cross-reference, which measurably closes the gap vs.
extracting blind from the transcript.

Note: model names change over time — if the default below is deprecated by
the time you read this, pass --model with a current one (check
https://platform.openai.com/docs/models).

Output:
    - Structured meeting notes (Markdown) printed to console
    - Saved to ./output/<timestamp>_<transcript filename stem>_notes.md
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; OPENAI_API_KEY can still be a real env var.

DEFAULT_MODEL = "gpt-4o-mini"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


SYSTEM_PROMPT = """You are an assistant that turns raw meeting transcripts into clear, \
structured meeting notes. The transcript comes from local speech-to-text and may \
contain transcription errors, especially on technical jargon and names — use \
surrounding context to infer the likely intended word where a mis-transcription \
is obvious, but do not invent details the transcript doesn't support.

This is a real multi-person meeting, not a monologue. Work to figure out who is \
speaking and attribute discussion points and action items to the correct named \
person:
- Track turn-taking cues like "next, X", "X, your turn", "thank you X, next is Y", \
  or a person referring to themselves ("next is me", "so I ..."). The speaker who \
  is currently talking owns whatever they say they'll do.
- Names are frequently mis-transcribed (e.g. a real name like "Heriz" might come \
  through as "Harris" or "Harry's"). If a name is used consistently for one \
  participant throughout the transcript, treat it as that person even if it's an \
  odd or unlikely-looking word — don't discard it. If the user-supplied participant \
  list below includes a plausible match for a garbled name, use the real name from \
  that list instead of the garbled transcript version.
- Only use "Unassigned" when the transcript genuinely gives no attribution signal \
  at all — prefer a best-effort attributed name over defaulting to Unassigned.

If the user supplies their own rough meeting notes below, treat them as a \
ground-truth cross-reference, not just extra color: prefer the notes' version of a \
name, decision, or action item's owner/deadline whenever the transcript is garbled, \
ambiguous, or contradicts them. Use the transcript to add detail and context the \
notes only summarized, and to catch anything the notes missed — but when the two \
genuinely conflict, the notes win.

Structure your response as Markdown with these sections (omit a section only \
if genuinely nothing in the transcript fits it):

## Summary
2-3 sentence overview of what the meeting covered.

## Discussion
Grouped by speaker (use their name as a subheading, e.g. "**James**"), a short \
bullet list of what that person said, raised, or reported — status updates, \
opinions, problems, proposals. Only include speakers who said something \
substantive.

## Decisions Made
Bullet list of concrete decisions the group reached, including process/workflow \
rules stated as decisions (e.g. branch/merge policy), not just feature decisions.

## Action Items
Bullet list formatted as "- [Owner]: Task (due: date if mentioned)". Attribute \
using the speaker-tracking guidance above; use "Unassigned" only as a last resort.

## Milestones
Bullet list of any dates, deadlines, or timeline/schedule items mentioned (e.g. \
"in two weeks: X", "due Friday: Y") that represent broader project milestones \
rather than individual action items. Omit this section if none were mentioned.

## Open Questions
Bullet list of unresolved items explicitly left for later.

Output only the Markdown notes — no preamble, no commentary about the transcript."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured meeting notes from a transcript using the OpenAI API."
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
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default=None,
        help='Comma-separated real names of meeting participants, e.g. "James, Heriz, '
        'Sham, Marcus, Aaron". Helps the model correctly attribute action items when '
        "Whisper has mis-transcribed names.",
    )
    parser.add_argument(
        "--notes-file",
        type=str,
        default=None,
        help="Path to a text file with your own rough meeting notes. Used as a "
        "ground-truth cross-reference for names/decisions/action items — closes "
        "most of the gap vs. extracting blind from a noisy transcript.",
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


def check_notes_file(notes_file_path: str | None) -> str | None:
    """Read an optional personal-notes file. Returns None if not given."""
    if not notes_file_path:
        return None
    path = Path(notes_file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        print(f"ERROR: Notes file not found: {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"ERROR: Notes file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return text


def import_openai():
    try:
        import openai
    except ImportError:
        print(
            "ERROR: The 'openai' package is not installed.\n"
            "Install dependencies with:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    return openai


def get_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Set it as an environment variable, or create a .env file in this\n"
            "folder containing:\n"
            "    OPENAI_API_KEY=your-key-here\n\n"
            "Get a key at https://platform.openai.com/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def summarize_transcript(
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    participants: str | None = None,
    notes_text: str | None = None,
) -> str:
    """
    Call the OpenAI API to turn a transcript into structured meeting notes.

    Returns the Markdown notes text. Exits the process on unrecoverable
    errors (missing package, missing/bad API key, API failure) — consistent
    with transcribe.py's error-handling style so both scripts behave the
    same whether run standalone or chained from pipeline.py.
    """
    openai = import_openai()
    api_key = get_api_key()
    client = openai.OpenAI(api_key=api_key)

    user_content = f"Transcript:\n\n{transcript_text}"
    if participants:
        user_content = (
            f"Known participants in this meeting: {participants}\n"
            f"(Use these real names to correct any garbled/mis-transcribed names you "
            f"encounter in the transcript below.)\n\n{user_content}"
        )
    if notes_text:
        user_content = (
            f"My own rough notes from this meeting (treat as ground truth — see "
            f"system instructions):\n\n{notes_text}\n\n{user_content}"
        )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except openai.AuthenticationError as exc:
        print(f"ERROR: OpenAI API authentication failed (check your API key): {exc}", file=sys.stderr)
        sys.exit(1)
    except openai.RateLimitError as exc:
        print(f"ERROR: OpenAI API rate limit hit: {exc}", file=sys.stderr)
        sys.exit(1)
    except openai.APIConnectionError as exc:
        print(f"ERROR: Could not reach the OpenAI API (network issue?): {exc}", file=sys.stderr)
        sys.exit(1)
    except openai.NotFoundError as exc:
        print(
            f"ERROR: OpenAI model '{model}' not found — it may be renamed/deprecated. "
            f"Check https://platform.openai.com/docs/models and pass --model: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except openai.APIStatusError as exc:
        print(f"ERROR: OpenAI API returned an error (status {exc.status_code}): {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - last-resort catch-all so we fail with a readable message
        print(f"ERROR: Summarization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    return response.choices[0].message.content.strip()


def summarize_and_save(
    transcript_text: str,
    base_name: str,
    model: str = DEFAULT_MODEL,
    participants: str | None = None,
    notes_text: str | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> dict:
    """
    Summarize transcript_text and save the result to
    <output_dir>/<timestamp>_<base_name>_notes.md. Prints the notes and timing.

    timestamp: pass one in to keep a transcript/notes pair aligned (e.g.
    from pipeline.py); otherwise one is generated here.

    Returns a dict with "notes", "output_path", "elapsed_seconds".
    """
    timestamp = timestamp or make_timestamp()
    output_dir = output_dir or (Path(__file__).parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{base_name}_notes.md"

    print(f"Generating structured meeting notes with OpenAI ({model})...")
    start_time = time.perf_counter()
    notes = summarize_transcript(
        transcript_text, model=model, participants=participants, notes_text=notes_text
    )
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
    notes_text = check_notes_file(args.notes_file)
    summarize_and_save(
        transcript_text,
        transcript_path.stem,
        model=args.model,
        participants=args.participants,
        notes_text=notes_text,
    )


if __name__ == "__main__":
    main()
