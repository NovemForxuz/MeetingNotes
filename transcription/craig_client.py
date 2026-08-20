#!/usr/bin/env python3
"""
craig_client.py — minimal async client for Craig's (craig.horse) recording
download API.

This API is undocumented publicly. Confirmed by making REAL requests
against a real Craig recording (not just reading source — an earlier
version of this file was built by reading Craig's open-source repo, which
turned out to be a stale/legacy snapshot that didn't match what's actually
deployed; that version's requests 404'd in real testing). Everything below
was verified working end-to-end against a live recording: base URL,
request/response shapes, the full running -> complete job transition, and
the final file download.

Important, deliberate limitation: Craig keeps a recording's id+key private
by design — DMed to whoever ran /join (or shown only to them as a fallback
if the DM fails). There is no way to obtain these automatically. This
client picks up from a URL/id+key a human already has (see
parse_recording_url()) and automates everything after that.

How a recording is fetched (confirmed against a real recording):
    1. GET  /api/v1/recordings/:id?key=...      -> {"recording": {...},
       "users": [...], "live": bool}. Used here just to fail fast with a
       clear error if the id/key are wrong, before triggering a job.
    2. POST /api/v1/recordings/:id/job?key=...  body:
       {"type": "recording", "options": {"format": "flac", "container": "zip"}}
       -> starts an async "job". format=flac/container=zip is deliberate,
       not Craig's default (there is no clear default here) — it's exactly
       the per-speaker FLAC zip structure transcribe_multitrack.py expects.
    3. GET  /api/v1/recordings/:id/job?key=...  -> poll:
       {"job": {"status": "running"|"complete"|..., "outputFileName": "...",
       ...}}. outputFileName appears immediately, even mid-job — status
       must reach "complete" before the file is actually ready, confirmed
       by polling a real job through its full running -> complete
       transition (encoding per track, then finalizing, then complete).
    4. GET  https://craig.horse/dl/<outputFileName>  -> the actual zip
       bytes. Confirmed: content-length matches the job's reported
       outputSize exactly.
"""

import asyncio
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp

BASE_URL = "https://craig.horse"
POLL_INTERVAL_SECONDS = 5
# A real 5-track, ~31-minute recording completed in well under a minute
# server-side; generous timeout regardless in case of a much longer meeting.
POLL_TIMEOUT_SECONDS = 900

# Job statuses observed in testing: "running" while in progress, "complete"
# when done. Treated as an allowlist rather than just checking != "running",
# so an unrecognized status fails loudly instead of polling forever.
JOB_STATUS_COMPLETE = "complete"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_ERROR_VALUES = {"error", "failed"}


class CraigError(Exception):
    """Any Craig API failure. Message is written to be safe/useful to show the user."""


# Cap on how much of a raw API response gets embedded in an error message. An
# unbounded dump (e.g. of a full HTML error page when a request hits the wrong
# domain/path) is enough to blow past Discord's message-length limit and cause a
# confusing secondary failure that masks the real error — confirmed the hard way.
RESPONSE_PREVIEW_LIMIT = 400


def _preview(data) -> str:
    text = str(data)
    if len(text) > RESPONSE_PREVIEW_LIMIT:
        return text[:RESPONSE_PREVIEW_LIMIT] + "... (truncated)"
    return text


def parse_recording_url(url_or_id: str, key: str | None = None) -> tuple:
    """
    Accept either a full Craig recording URL
    (https://craig.horse/rec/<id>?key=<key>, as Craig DMs/shows the user)
    or a bare recording id with a separately-supplied key.

    Returns (id, key). Raises CraigError with a user-facing message if the
    URL doesn't look right.
    """
    text = url_or_id.strip()
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            raise CraigError(f"Couldn't find a recording ID in that URL: {url_or_id}")
        rec_id = parts[-1]
        qs = parse_qs(parsed.query)
        rec_key = (qs.get("key") or [None])[0]
        if not rec_key:
            raise CraigError(
                "That URL doesn't have a `key` in it — paste the full link Craig gave "
                "you, including the `?key=...` part."
            )
        return rec_id, rec_key

    if not key:
        raise CraigError("A bare recording ID was given without a key.")
    return text, key


async def _get_json(session: aiohttp.ClientSession, url: str, **kwargs) -> tuple:
    """GET url, returning (status, parsed_json_or_raw_text_fallback)."""
    async with session.get(url, **kwargs) as resp:
        try:
            return resp.status, await resp.json()
        except aiohttp.ContentTypeError:
            return resp.status, {"raw": await resp.text()}


async def fetch_recording_info(session: aiohttp.ClientSession, rec_id: str, key: str) -> dict:
    """Fail fast with a clear error if the id/key are wrong, before triggering a job."""
    status, data = await _get_json(session, f"{BASE_URL}/api/v1/recordings/{rec_id}", params={"key": key})
    if status != 200 or "recording" not in data:
        raise CraigError(
            f"Craig rejected that recording ID/key (HTTP {status}). This usually means "
            f"the link has expired (Craig keeps recordings 7 days) or was typed wrong. "
            f"Raw response: {_preview(data)}"
        )
    return data


async def start_job(session: aiohttp.ClientSession, rec_id: str, key: str) -> None:
    async with session.post(
        f"{BASE_URL}/api/v1/recordings/{rec_id}/job",
        params={"key": key},
        json={"type": "recording", "options": {"format": "flac", "container": "zip"}},
    ) as resp:
        try:
            data = await resp.json()
        except aiohttp.ContentTypeError:
            data = {"raw": await resp.text()}
        if resp.status != 200 or "job" not in data:
            raise CraigError(
                f"Craig refused to start cooking the recording (HTTP {resp.status}). "
                f"Raw response: {_preview(data)}"
            )


def estimate_cook_progress(job: dict, total_tracks: int | None = None) -> tuple:
    """
    Best-effort (fraction 0-1, short label) from a job status response.

    Based on real observed shapes: state.type moves "starting" ->
    "encoding" (with a per-track state.tracks[n].progress 0-100, tracks not
    yet started simply absent from the dict) -> "finalizing" -> status
    becomes "complete". There's no single overall-progress field, so this
    is a reasonable approximation, not an exact number from Craig itself.
    """
    state = job.get("state") or {}
    state_type = state.get("type")

    if job.get("status") == JOB_STATUS_COMPLETE:
        return 1.0, "complete"
    if state_type == "finalizing":
        return 0.95, "finalizing"
    if state_type == "encoding":
        tracks = state.get("tracks") or {}
        n = total_tracks or max(len(tracks), 1)
        track_progress = sum(t.get("progress", 0) for t in tracks.values()) / 100
        fraction = min(track_progress / n, 0.94)  # cap below "finalizing"'s 0.95
        done = sum(1 for t in tracks.values() if t.get("progress", 0) >= 100)
        return fraction, f"encoding track {min(done + 1, n)}/{n}"
    if state_type == "starting":
        return 0.0, "starting"
    return 0.0, state_type or "running"


async def wait_for_job(
    session: aiohttp.ClientSession,
    rec_id: str,
    key: str,
    total_tracks: int | None = None,
    on_progress=None,
) -> str:
    """
    Poll until the job reaches status 'complete'. Returns the output filename.

    on_progress, if given, is awaited on every poll as
    on_progress(fraction: float, label: str) — see estimate_cook_progress().
    """
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        status_code, data = await _get_json(
            session, f"{BASE_URL}/api/v1/recordings/{rec_id}/job", params={"key": key}
        )
        job = data.get("job") if isinstance(data, dict) else None
        if status_code != 200 or job is None:
            raise CraigError(
                f"Craig reported an error while cooking (HTTP {status_code}). "
                f"Raw response: {_preview(data)}"
            )

        job_status = job.get("status")
        if job_status in JOB_STATUS_ERROR_VALUES:
            raise CraigError(f"Craig's cook job failed (status={job_status}): {_preview(job)}")
        if job_status == JOB_STATUS_COMPLETE:
            filename = job.get("outputFileName")
            if not filename:
                raise CraigError(f"Job is complete but Craig gave no output filename: {_preview(job)}")
            if on_progress:
                await on_progress(*estimate_cook_progress(job, total_tracks))
            return filename
        if job_status != JOB_STATUS_RUNNING:
            raise CraigError(f"Unrecognized job status from Craig: {_preview(job)}")

        if on_progress:
            await on_progress(*estimate_cook_progress(job, total_tracks))

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    raise CraigError(
        f"Timed out after {POLL_TIMEOUT_SECONDS}s waiting for Craig to finish cooking the recording."
    )


async def download_cooked_file(session: aiohttp.ClientSession, filename: str, dest_path: Path) -> Path:
    async with session.get(f"{BASE_URL}/dl/{filename}") as resp:
        if resp.status != 200:
            raise CraigError(f"Failed to download the cooked recording (HTTP {resp.status}): {filename}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(1 << 16):
                f.write(chunk)
    return dest_path


def extract_craig_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a cooked Craig zip (per-speaker FLACs + info.txt) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


async def fetch_and_extract_recording(
    url_or_id: str, dest_dir: Path, key: str | None = None, on_progress=None
) -> Path:
    """
    End-to-end: parse the recording URL, trigger cooking, wait for it,
    download, and extract into dest_dir (ready for transcribe_multitrack.py
    / pipeline.py). Returns dest_dir.

    on_progress, if given, is awaited with (fraction: float, label: str) on
    every poll during cooking (see estimate_cook_progress() for what the
    fraction/label mean) and once more, with (1.0, "downloaded"), after the
    file download completes.
    """
    rec_id, rec_key = parse_recording_url(url_or_id, key)
    async with aiohttp.ClientSession() as session:
        info = await fetch_recording_info(session, rec_id, rec_key)
        total_tracks = len(info.get("users", [])) or None
        await start_job(session, rec_id, rec_key)
        filename = await wait_for_job(session, rec_id, rec_key, total_tracks, on_progress)
        zip_path = dest_dir.parent / f"_craig_{rec_id}.zip"
        await download_cooked_file(session, filename, zip_path)
        if on_progress:
            await on_progress(1.0, "downloaded")
        extract_craig_zip(zip_path, dest_dir)
        zip_path.unlink(missing_ok=True)
    return dest_dir
