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

Output:
    - Transcript printed to console
    - Transcript saved to ./output/<input_filename>.txt

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
from pathlib import Path

# Models ordered roughly from fastest/least-accurate to slowest/most-accurate.
VALID_MODELS = ["tiny", "base", "small", "medium", "large"]

SUPPORTED_EXTENSIONS = {
    ".flac", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file locally using OpenAI Whisper (offline, no API)."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file to transcribe (e.g. a .flac recording from Discord).",
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


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path).expanduser().resolve()

    check_ffmpeg()
    check_audio_file(audio_path)
    whisper = import_whisper()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.txt"

    device = detect_device()

    print(f"Loading Whisper model '{args.model}'...")
    try:
        model = whisper.load_model(args.model, device=device)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        print(f"ERROR: Failed to load Whisper model '{args.model}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribing '{audio_path.name}'... (this may take a while on CPU)")
    start_time = time.perf_counter()
    try:
        result = model.transcribe(
            str(audio_path),
            language=args.language,
            initial_prompt=args.initial_prompt,
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


if __name__ == "__main__":
    main()
