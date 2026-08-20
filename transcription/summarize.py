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
    python summarize.py path\to\transcript.txt --single-pass   # faster/cheaper, more prone to dropping content

By default this runs a three-pass pipeline (extract everything exhaustively,
organize it, then verify the draft against the raw extraction and backfill
anything missing) specifically to avoid the failure mode where asking a
model to summarize a long transcript in one shot quietly drops minor-seeming
details (a single throwaway line, a decision after a tangent). Confirmed in
testing that even the organizing pass alone can still drop real content
despite explicit instructions not to — the verification pass exists because
of that. Pass --single-pass to skip all of this for one direct call instead
— much faster and cheaper, but confirmed to drop real content on a long
transcript. See README.md ("Avoiding dropped content") for details.

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
    return datetime.now().strftime("%Y%m%d_%H%M")


# --- Pass 1: exhaustive extraction (favor completeness over organization) ---
EXTRACT_SYSTEM_PROMPT = """You are performing the FIRST pass of a three-pass meeting-notes \
pipeline: exhaustive extraction. Do not organize, summarize, or polish anything yet — \
later passes handle that. Your only job right now is making sure nothing gets missed.

Go through the ENTIRE transcript from start to finish, in order, and extract every \
distinct point as its own bullet: status updates, decisions, action items, deadlines/\
dates, questions raised, and any other notable statement. Even a short, single-line \
remark ("I'll start tonight") deserves its own bullet. Do not skip ahead to only the \
parts that seem important — work through the whole transcript, including anything \
after a tangent or unrelated aside.

Attribute each bullet to whoever said it:
- If a line already starts with an explicit speaker label (e.g. "[00:12] Name: ..."), \
  that label is ground truth from the recording setup — use it directly.
- Otherwise infer from context: turn-calling ("next, X"), self-reference ("so I..."), \
  being addressed by name, or a point being replied to.
- Names are frequently mis-transcribed into an unrelated but phonetically similar \
  word. If the participant list below includes a plausible match for a garbled name, \
  use the real name instead of the garbled transcript version.

Rules:
- Err heavily on the side of including too much. Omitting something that turns out to \
  matter is worse than including something minor or repetitive.
- Do not deduplicate, merge similar points, or reorganize by topic — keep the original \
  chronological order. That's the next pass's job, not yours.
- If the user's own rough notes are supplied below, still extract independently from \
  the full transcript in addition to them — the notes are for cross-referencing in the \
  next pass, not a shortcut that lets you skip transcript coverage now.

Output a flat, chronological bullet list only, each formatted as "- [Speaker] point." \
No headers, no sections, no commentary, no summarizing — just the raw extraction."""


# --- Pass 2: organize the raw extraction into the final structured notes ---
STRUCTURE_SYSTEM_PROMPT = """You are performing the SECOND pass of a three-pass \
meeting-notes pipeline: organizing an already-exhaustive raw extraction into clean, \
structured notes. A first pass already went through the full transcript and pulled \
out every point as a flat bullet list — your job is to organize it, not re-extract \
or drop anything. A third pass will check your draft against the raw extraction \
afterward, but treat this as your one real chance to get it right, not a safety net.

Critical rule: every bullet in the raw extraction below must be reflected somewhere \
in your output. You may merge two bullets only if they state the exact same fact \
twice (true duplicates) — never drop a bullet just because it seems minor, \
repetitive-but-distinct, or hard to categorize. If something doesn't fit an existing \
section cleanly, put it in the closest one rather than omitting it.

If the user's own rough notes are supplied below, use them to resolve ambiguity \
(correct a name, clarify an owner/deadline) — but they should never cause you to drop \
anything the raw extraction captured.

Structure your response as Markdown with these sections (omit a section only if \
genuinely nothing fits it):

## Summary
2-3 sentence overview of what the meeting covered.

## Discussion
Grouped by speaker (use their name as a subheading, e.g. "**James**"), a bullet list \
of what that person said, raised, or reported — status updates, opinions, problems, \
proposals. Only include speakers who said something substantive.

## Decisions Made
Bullet list of concrete decisions the group reached, including process/workflow \
rules stated as decisions (e.g. branch/merge policy), not just feature decisions.

## Action Items
Bullet list formatted as "- [Owner]: Task (due: date if mentioned)". Use \
"Unassigned" only as a last resort, when the raw extraction genuinely gives no \
attribution signal at all.

## Milestones
Bullet list of any dates, deadlines, or timeline/schedule items mentioned (e.g. \
"in two weeks: X", "due Friday: Y") that represent broader project milestones \
rather than individual action items. Omit this section if none were mentioned.

## Open Questions
Bullet list of unresolved items explicitly left for later.

Output only the Markdown notes — no preamble, no commentary."""


# --- Pass 3: verify the draft against the raw extraction and backfill anything missing ---
VERIFY_SYSTEM_PROMPT = """You are performing the FINAL pass of a three-pass meeting-notes \
pipeline: verification. Passes 1 and 2 already produced (1) a raw, exhaustive bullet-point \
extraction from the full transcript, and (2) a draft of structured meeting notes organized \
from that extraction. Even with explicit instructions not to, pass 2 sometimes still \
compresses away real content when condensing hundreds of raw bullets into a clean \
document — your job is to catch and fix that.

Check every bullet in the raw extraction below against the draft. If a bullet's content is \
not reflected anywhere in the draft — even implicitly or in summarized form — add it to \
whichever existing section fits best. Do not remove, shorten, or rewrite anything already \
correctly present in the draft; only ADD what's missing. If the draft already fully covers \
the raw extraction, return it unchanged.

Keep the same Markdown structure and section headers as the draft. Output only the final, \
complete Markdown notes — no preamble, no commentary about what you changed."""


# --- Single-pass mode (--single-pass): faster/cheaper, more prone to dropping content ---
SINGLE_PASS_SYSTEM_PROMPT = """You are an assistant that turns raw meeting transcripts \
into clear, structured meeting notes. The transcript comes from local speech-to-text \
and may contain transcription errors, especially on technical jargon and names — use \
surrounding context to infer the likely intended word where a mis-transcription is \
obvious, but do not invent details the transcript doesn't support.

This is a real multi-person meeting, not a monologue. Work to figure out who is \
speaking and attribute discussion points and action items to the correct named \
person:
- If a line already starts with an explicit speaker label (e.g. "[00:12] Name: ..."), \
  that label is ground truth from the recording setup, not a guess — trust it \
  directly rather than re-inferring the speaker from context.
- Otherwise, track whatever attribution signals the transcript actually gives: explicit \
  turn-calling ("next, X", "X, your turn"), a person being directly addressed by \
  name, someone referring to themselves ("so I...", "my part is..."), or a named \
  person's point being replied to or built on. Not every meeting is a structured \
  round-robin — a freeform discussion may only signal who's speaking through being \
  addressed by name or context; use whatever signal is present rather than only \
  looking for round-robin cues.
- Names are frequently mis-transcribed into an unrelated but phonetically similar \
  real word (a name could come through as a common noun or a different name \
  entirely). If an odd or unlikely-looking word is used consistently in place of a \
  person throughout the transcript, treat it as a mis-transcribed name rather than \
  discarding it. If the user-supplied participant list below includes a plausible \
  match for a garbled name, use the real name from that list instead of the \
  garbled transcript version.
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
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Skip the three-pass extract/organize/verify pipeline and do one direct "
        "call instead. Faster/cheaper, but confirmed in testing to drop real content "
        "(a single throwaway line, a decision after a tangent) on long transcripts.",
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


def _call_openai(
    client,
    openai,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> str:
    """
    Shared OpenAI chat-completion call with consistent error handling.
    Exits the process on unrecoverable errors, same style as transcribe.py.
    Temperature is fixed low (0.2) — this is factual extraction/organization,
    not creative writing, and a lower temperature reduces the model's
    tendency to paraphrase-and-compress rather than preserve detail.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
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

    choice = response.choices[0]
    if choice.finish_reason == "length":
        # This is exactly the silent-content-loss failure mode this multi-pass design
        # exists to avoid — surface it loudly instead of returning a truncated
        # result as if it were complete.
        print(
            f"WARNING: OpenAI response was cut off by the max_tokens limit ({max_tokens}) "
            f"mid-output — this response is INCOMPLETE, not just short. Content past the "
            f"cutoff is silently missing. Increase max_tokens for this call.",
            file=sys.stderr,
        )

    return choice.message.content.strip()


def _build_context_prefix(participants: str | None, notes_text: str | None) -> str:
    prefix = ""
    if participants:
        prefix += (
            f"Known participants in this meeting: {participants}\n"
            f"(Use these real names to correct any garbled/mis-transcribed names you "
            f"encounter below.)\n\n"
        )
    if notes_text:
        prefix += (
            f"My own rough notes from this meeting (treat as ground truth — see "
            f"system instructions):\n\n{notes_text}\n\n"
        )
    return prefix


def summarize_transcript(
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    participants: str | None = None,
    notes_text: str | None = None,
    single_pass: bool = False,
) -> str:
    """
    Turn a transcript into structured meeting notes via the OpenAI API.

    By default runs a three-pass pipeline (exhaustive extraction, organizing,
    then verifying the draft against the raw extraction and backfilling
    anything missing) specifically to avoid the failure mode where a
    single-shot summarize call quietly drops minor-seeming content on a
    long transcript — confirmed in testing that even the organizing pass
    alone can still drop real content despite explicit instructions not
    to, which is why the verification pass exists. Pass single_pass=True
    for the old one-call behavior — faster/cheaper, more prone to dropping
    content.

    Returns the Markdown notes text. Exits the process on unrecoverable
    errors (missing package, missing/bad API key, API failure) — consistent
    with transcribe.py's error-handling style so both scripts behave the
    same whether run standalone or chained from pipeline.py.
    """
    openai = import_openai()
    api_key = get_api_key()
    client = openai.OpenAI(api_key=api_key)

    context_prefix = _build_context_prefix(participants, notes_text)

    if single_pass:
        user_content = context_prefix + f"Transcript:\n\n{transcript_text}"
        return _call_openai(
            client, openai, model, SINGLE_PASS_SYSTEM_PROMPT, user_content, max_tokens=2000
        )

    # Sized generously for long transcripts: a dense ~30-minute, 5-person meeting
    # produced ~470 extraction bullets in testing (~9-10K tokens). Extraction
    # runs roughly one bullet per transcript line, so scale with transcript
    # length rather than assume a fixed size — better to over-allocate than
    # silently truncate (see the finish_reason=="length" check in _call_openai).
    estimated_lines = max(transcript_text.count("\n") + 1, 1)
    extract_max_tokens = min(max(estimated_lines * 25, 4000), 16000)

    print(f"  Pass 1/3: extracting exhaustively (max_tokens={extract_max_tokens})...")
    extract_user_content = context_prefix + f"Transcript:\n\n{transcript_text}"
    raw_extraction = _call_openai(
        client, openai, model, EXTRACT_SYSTEM_PROMPT, extract_user_content,
        max_tokens=extract_max_tokens,
    )

    print("  Pass 2/3: organizing into structured notes...")
    structure_user_content = context_prefix + f"Raw extraction:\n\n{raw_extraction}"
    draft_notes = _call_openai(
        client, openai, model, STRUCTURE_SYSTEM_PROMPT, structure_user_content, max_tokens=6000
    )

    # Confirmed in testing: pass 2 can still compress away real content even with
    # explicit "never drop a bullet" instructions, when condensing hundreds of raw
    # bullets into a clean document. This pass diffs the draft against the raw
    # extraction and backfills anything missing, rather than trusting pass 2 blind.
    print("  Pass 3/3: verifying draft against raw extraction, backfilling gaps...")
    verify_user_content = (
        f"Raw extraction:\n\n{raw_extraction}\n\nDraft structured notes:\n\n{draft_notes}"
    )
    return _call_openai(
        client, openai, model, VERIFY_SYSTEM_PROMPT, verify_user_content, max_tokens=6000
    )


def summarize_and_save(
    transcript_text: str,
    base_name: str,
    model: str = DEFAULT_MODEL,
    participants: str | None = None,
    notes_text: str | None = None,
    single_pass: bool = False,
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

    mode = "single-pass" if single_pass else "three-pass (extract, organize, verify)"
    print(f"Generating structured meeting notes with OpenAI ({model}, {mode})...")
    start_time = time.perf_counter()
    notes = summarize_transcript(
        transcript_text,
        model=model,
        participants=participants,
        notes_text=notes_text,
        single_pass=single_pass,
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
        single_pass=args.single_pass,
    )


if __name__ == "__main__":
    main()
