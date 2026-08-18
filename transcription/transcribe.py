#!/usr/bin/env python3
"""
transcribe.py — MVP local transcription script for MeetingNotes.

Transcribes an audio file to text using OpenAI's open-source Whisper model,
running fully offline (no API calls). Intended as the first building block
of the MeetingNotes pipeline: Discord audio -> transcript -> (later) Claude
API for structured meeting notes.

Usage:
    python transcribe.py path\to\recording.flac
    python transcribe.py path\to\recording.flac --model small
    python transcribe.py path\to\recording.flac --model small --initial-prompt "Docker, CI/CD, ASP.NET Core, C#, .NET, Git, MVP, webhooks, containers"
    python transcribe.py                          # auto-picks the one file in ./input/
    python transcribe.py recording.flac            # resolves against ./input/ if not found as-is

Output:
    - Transcript printed to console
    - Transcript saved to ./output/<timestamp>_<input_filename>.txt

Notes on accuracy:
    - The "base" model is fast but weak on domain-specific jargon (e.g. it
      commonly mis-hears "Docker" as "darker", "containers" as "condoms").
      Try --model small or --model medium if technical terms matter.
    - Use --initial-prompt to bias Whisper toward expected vocabulary (names,
      acronyms, product/tech terms). This measurably improves recognition of
      terms that appear in the prompt.
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Models ordered roughly from fastest/least-accurate to slowest/most-accurate.
VALID_MODELS = ["tiny", "base", "small", "medium", "large"]

SUPPORTED_EXTENSIONS = {
    ".flac", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga",
}

INPUT_DIR = Path(__file__).parent / "input"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_audio_path(audio_arg: str | None) -> Path:
    """
    Figure out which audio file to transcribe.

    - No argument given: look in ./input/ for exactly one supported audio
      file and use it. Errors (with a clear message) if there are zero or
      more than one.
    - Argument given: use it as-is (absolute or relative to cwd) if it
      exists; otherwise fall back to treating it as a filename inside
      ./input/, so you can drop a file in ./input/ and just pass its name.
    """
    if audio_arg is None:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            print(
                f"ERROR: No audio file given, and no supported audio file found in "
                f"{INPUT_DIR}.\nEither pass a path, or drop a file into that folder.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(candidates) > 1:
            names = "\n".join(f"  - {p.name}" for p in candidates)
            print(
                f"ERROR: Multiple audio files found in {INPUT_DIR}, and no path was given "
                f"to disambiguate:\n{names}\nPass one explicitly, e.g. "
                f"transcribe.py {candidates[0].name}",
                file=sys.stderr,
            )
            sys.exit(1)
        return candidates[0].resolve()

    given_path = Path(audio_arg).expanduser()
    if given_path.exists():
        return given_path.resolve()

    fallback_path = INPUT_DIR / audio_arg
    if fallback_path.exists():
        return fallback_path.resolve()

    # Neither exists — return the as-given path so the normal "not found"
    # error message downstream reports the path the user actually typed.
    return given_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file locally using OpenAI Whisper (offline, no API)."
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
        choices=VALID_MODELS,
        help="Whisper model size to use (default: base). Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force a language code (e.g. 'en') instead of Whisper auto-detecting it. "
        "Useful if auto-detection drifts mid-file.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Optional text to prime Whisper with expected vocabulary (names, acronyms, "
        "technical terms). E.g. \"Docker, CI/CD, ASP.NET Core, C#, .NET, Git, MVP\". "
        "Improves recognition of jargon that base/small models otherwise mis-hear.",
    )
    return parser.parse_args()


def check_ffmpeg() -> None:
    """Whisper shells out to ffmpeg to decode audio; fail early with a clear message if it's missing."""
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg was not found on your PATH.\n"
            "Whisper requires ffmpeg to decode audio files.\n\n"
            "On Windows, install it with one of:\n"
            "    winget install Gyan.FFmpeg\n"
            "    choco install ffmpeg\n"
            "  or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its\n"
            "  bin\\ folder to your PATH, then restart your terminal.\n\n"
            "See README.md for more details.",
            file=sys.stderr,
        )
        sys.exit(1)


def check_audio_file(audio_path: Path) -> None:
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if not audio_path.is_file():
        print(f"ERROR: Path is not a file: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"WARNING: '{audio_path.suffix}' is not a commonly supported audio format.\n"
            f"Whisper (via ffmpeg) usually handles: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.\n"
            "Attempting to transcribe anyway...",
            file=sys.stderr,
        )


def import_whisper():
    try:
        import whisper  # noqa: WPS433 (intentional lazy import for clearer error handling)
    except ImportError:
        print(
            "ERROR: The 'openai-whisper' package is not installed.\n"
            "Install dependencies with:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    return whisper


def detect_device() -> str:
    """Use CUDA automatically if available, otherwise fall back to CPU."""
    try:
        import torch
    except ImportError:
        print("Using device: cpu (torch not importable)")
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({device_name})")
        return "cuda"

    print(
        "Using device: cpu (no CUDA GPU detected, or torch was installed without CUDA "
        "support — see README.md for enabling GPU acceleration)"
    )
    return "cpu"


def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    language: str | None = None,
    initial_prompt: str | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> dict:
    """
    Run the full local transcription pipeline for a single audio file.

    Validates ffmpeg/the input file, loads the Whisper model on the best
    available device, transcribes, saves the transcript to
    <output_dir>/<timestamp>_<audio filename stem>.txt, prints
    progress/results, and returns a dict with the outcome. Exits the
    process (sys.exit) on unrecoverable errors — this is intentional so
    callers (this script's CLI, or pipeline.py chaining into summarize.py)
    all fail the same way without needing their own try/except around this
    call.

    timestamp: pass one in to keep a transcript/notes pair aligned (e.g.
    from pipeline.py); otherwise one is generated here.

    Returns:
        {
            "transcript": str,
            "output_path": Path,
            "language": str,
            "device": str,
            "elapsed_seconds": float,
            "timestamp": str,
        }
    """
    check_ffmpeg()
    check_audio_file(audio_path)
    whisper = import_whisper()

    timestamp = timestamp or make_timestamp()
    output_dir = output_dir or (Path(__file__).parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{audio_path.stem}.txt"

    device = detect_device()

    print(f"Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        print(f"ERROR: Failed to load Whisper model '{model_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribing '{audio_path.name}'... (this may take a while on CPU)")
    start_time = time.perf_counter()
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=initial_prompt,
        )
    except FileNotFoundError as exc:
        # Typically means ffmpeg couldn't be invoked despite passing the PATH check above.
        print(f"ERROR: Could not read audio file (ffmpeg issue?): {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"ERROR: Whisper failed to process the audio file (unsupported/corrupt format?): {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - last-resort catch-all so we fail with a readable message
        print(f"ERROR: Transcription failed: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - start_time

    transcript = result["text"].strip()
    detected_language = result.get("language", "unknown")

    output_path.write_text(transcript, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)
    print(f"\nSaved transcript to: {output_path}")
    print(f"Detected language: {detected_language}")
    print(f"Transcription took {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes).")

    return {
        "transcript": transcript,
        "output_path": output_path,
        "language": detected_language,
        "device": device,
        "elapsed_seconds": elapsed,
        "timestamp": timestamp,
    }


def main() -> None:
    args = parse_args()
    audio_path = resolve_audio_path(args.audio_path)
    transcribe_audio(
        audio_path,
        model_name=args.model,
        language=args.language,
        initial_prompt=args.initial_prompt,
    )


if __name__ == "__main__":
    main()
