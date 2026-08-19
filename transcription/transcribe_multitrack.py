#!/usr/bin/env python3
"""
transcribe_multitrack.py — transcribe a Craig (craig.chat) multi-track
Discord recording into one chronological, speaker-labeled transcript.

Craig records each Discord speaker to a separate audio track instead of one
mixed-down file. That sidesteps the biggest limitation of transcribe.py: on
a single mixed file, Whisper has to *guess* who's talking from context; with
per-speaker tracks, we know who said what for free, before any guessing.

How it works:
    1. Find each speaker's audio file in the given folder (Craig names them
       "<track#>-<discord username>.<ext>", per info.txt).
    2. Transcribe each track separately with Whisper, keeping per-segment
       timestamps (not just the flattened text transcribe.py returns).
    3. Filter out segments Whisper produced during silence/near-silence on
       that track (a well-known Whisper failure mode: it can hallucinate
       plausible-sounding text over silence — much more of an issue here
       since every track spans the full meeting, most of which is silence
       for any single speaker).
    4. Merge every kept segment from every track into one list sorted by
       start time, formatted as "[MM:SS] Speaker: text".

Usage:
    python transcribe_multitrack.py path\to\craig-export-folder
    python transcribe_multitrack.py path\to\craig-export-folder --model small --language en
    python transcribe_multitrack.py path\to\craig-export-folder --name-map "novemforxuz=Heriz,shamgoh=Sham"

The output is a normal transcript .txt — feed it into summarize.py exactly
like transcribe.py's output. Since speakers are already labeled, you likely
won't need --participants there anymore (though --notes-file is still
worth using — see transcription/README.md).

Output:
    - Merged transcript printed to console
    - Saved to ./output/<timestamp>_<recording name>_multitrack.txt

Note: transcribing N tracks takes roughly N times as long as transcribing
one mixed file at the same model size, since each track is transcribed
independently.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import transcribe

TRACK_FILENAME_RE = re.compile(r"^(\d+)-(.+)$")

# Whisper segments with a no_speech_prob at or above this are treated as
# silence/hallucination and dropped rather than included in the merged
# transcript. 0.6 matches Whisper's own default no_speech_threshold.
NO_SPEECH_THRESHOLD = 0.6

# Segments with an avg_logprob below this are treated as low-confidence
# hallucination and dropped too. no_speech_prob alone isn't enough — on real
# multi-track audio (breathing, mic noise, background sound that isn't
# silence but also isn't intelligible speech), Whisper can report low
# no_speech_prob while still inventing text with very low confidence.
# Calibrated against real output: garbage segments measured around -4.8,
# genuine (if imperfect) speech around -0.6 to -0.9.
AVG_LOGPROB_THRESHOLD = -1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a Craig multi-track Discord export into one "
        "chronological, speaker-labeled transcript."
    )
    parser.add_argument(
        "craig_dir",
        type=str,
        help="Path to the extracted Craig export folder (contains info.txt and one "
        "audio file per speaker).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=transcribe.VALID_MODELS,
        help="Whisper model size to use (default: base), applied to every track.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force a language code (e.g. 'en') instead of Whisper auto-detecting it "
        "per track.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Vocabulary hint text passed to Whisper for every track (see transcribe.py).",
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help='Comma-separated "discord_username=Real Name" pairs to relabel speakers, '
        'e.g. "novemforxuz=Heriz,shamgoh=Sham". Usernames not listed keep their '
        "Discord username as the label.",
    )
    return parser.parse_args()


def parse_name_map(name_map_arg: str | None) -> dict:
    if not name_map_arg:
        return {}
    mapping = {}
    for pair in name_map_arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(
                f"WARNING: Ignoring malformed --name-map entry (expected username=Name): "
                f"'{pair}'",
                file=sys.stderr,
            )
            continue
        username, real_name = pair.split("=", 1)
        mapping[username.strip()] = real_name.strip()
    return mapping


def discover_tracks(craig_dir: Path) -> list:
    """
    Find per-speaker audio files in a Craig export folder.

    Craig names them "<track#>-<discord username>.<ext>" (per info.txt).
    Returns a list of (track_number, username, path) tuples sorted by
    track number. Exits with a clear error if none are found.
    """
    if not craig_dir.exists() or not craig_dir.is_dir():
        print(f"ERROR: Not a folder: {craig_dir}", file=sys.stderr)
        sys.exit(1)

    tracks = []
    for path in craig_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in transcribe.SUPPORTED_EXTENSIONS:
            continue
        match = TRACK_FILENAME_RE.match(path.stem)
        if not match:
            continue
        track_number, username = match.groups()
        tracks.append((int(track_number), username, path))

    if not tracks:
        print(
            f"ERROR: No per-speaker audio files found in {craig_dir}.\n"
            f"Expected files named like '1-username.flac' (check info.txt in that "
            f"folder for the track list Craig recorded).",
            file=sys.stderr,
        )
        sys.exit(1)

    tracks.sort(key=lambda t: t[0])
    return tracks


def get_recording_name(craig_dir: Path) -> str:
    """Prefer the recording name from info.txt ("Recording <name>"); fall back to the folder name."""
    info_path = craig_dir / "info.txt"
    if info_path.exists():
        try:
            first_line = info_path.read_text(encoding="utf-8").splitlines()[0]
            match = re.match(r"Recording\s+(\S+)", first_line)
            if match:
                return match.group(1)
        except (OSError, IndexError):
            pass
    return craig_dir.stem


def transcribe_track_segments(
    model,
    track_path: Path,
    language: str | None,
    initial_prompt: str | None,
) -> list:
    """
    Transcribe one track, returning kept (start, end, text) segments with
    silence/hallucination-prone segments filtered out.
    """
    try:
        result = model.transcribe(
            str(track_path),
            language=language,
            initial_prompt=initial_prompt,
        )
    except Exception as exc:  # noqa: BLE001 - surface failure, but keep going with other tracks
        print(f"WARNING: Failed to transcribe {track_path.name}: {exc}", file=sys.stderr)
        return []

    kept = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if seg.get("no_speech_prob", 0.0) >= NO_SPEECH_THRESHOLD:
            continue
        if seg.get("avg_logprob", 0.0) < AVG_LOGPROB_THRESHOLD:
            continue
        kept.append((seg["start"], seg["end"], text))
    return kept


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def transcribe_multitrack(
    craig_dir: Path,
    model_name: str = "base",
    language: str | None = None,
    initial_prompt: str | None = None,
    name_map: dict | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> dict:
    """
    Transcribe every track in a Craig export folder and merge them into one
    chronological, speaker-labeled transcript. Exits the process on
    unrecoverable errors, same style as transcribe.py.

    Returns:
        {
            "transcript": str,
            "output_path": Path,
            "device": str,
            "elapsed_seconds": float,
            "timestamp": str,
            "track_count": int,
        }
    """
    name_map = name_map or {}
    transcribe.check_ffmpeg()
    tracks = discover_tracks(craig_dir)
    whisper = transcribe.import_whisper()

    timestamp = timestamp or transcribe.make_timestamp()
    output_dir = output_dir or (Path(__file__).parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_name = get_recording_name(craig_dir)
    output_path = output_dir / f"{timestamp}_{recording_name}_multitrack.txt"

    device = transcribe.detect_device()

    print(f"Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to load Whisper model '{model_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tracks)} track(s): " + ", ".join(f"{n}-{u}" for n, u, _ in tracks))

    all_segments = []  # (start, end, speaker_label, text)
    start_time = time.perf_counter()
    for i, (track_number, username, track_path) in enumerate(tracks, start=1):
        speaker_label = name_map.get(username, username)
        print(f"Transcribing track {i}/{len(tracks)}: {track_path.name} ({speaker_label})...")
        segments = transcribe_track_segments(model, track_path, language, initial_prompt)
        print(f"  -> {len(segments)} segment(s) kept (silence/low-confidence filtered out)")
        for seg_start, seg_end, text in segments:
            all_segments.append((seg_start, seg_end, speaker_label, text))
    elapsed = time.perf_counter() - start_time

    all_segments.sort(key=lambda s: s[0])
    lines = [
        f"[{format_timestamp(start)}] {speaker}: {text}"
        for start, end, speaker, text in all_segments
    ]
    transcript = "\n".join(lines)

    output_path.write_text(transcript, encoding="utf-8")

    print("\n" + "=" * 60)
    print("MERGED TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)
    print(f"\nSaved merged transcript to: {output_path}")
    print(f"Transcribed {len(tracks)} tracks in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes).")

    return {
        "transcript": transcript,
        "output_path": output_path,
        "device": device,
        "elapsed_seconds": elapsed,
        "timestamp": timestamp,
        "track_count": len(tracks),
    }


def main() -> None:
    args = parse_args()
    craig_dir = Path(args.craig_dir).expanduser().resolve()
    name_map = parse_name_map(args.name_map)
    transcribe_multitrack(
        craig_dir,
        model_name=args.model,
        language=args.language,
        initial_prompt=args.initial_prompt,
        name_map=name_map,
    )


if __name__ == "__main__":
    main()
