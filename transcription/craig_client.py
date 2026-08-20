#!/usr/bin/env python3
"""
craig_client.py — minimal async client for Craig's (craig.chat) recording
download API.

This API is undocumented publicly. Everything here was reverse-engineered
by reading Craig's own open-source code directly
(https://github.com/CraigChat/craig — apps/download/api and apps/download/page),
not from any published API docs, because none exist. It has NOT been
exercised against a real live recording yet — only verified by reading the
source. Treat the first real run as a live test, not a proven path; if
something doesn't match reality, the error messages here are written to
surface the raw response so a mismatch is debuggable rather than mysterious.

Important, deliberate limitation: Craig keeps a recording's id+key private
by design — DMed to whoever ran /join (or shown only to them as a fallback
if the DM fails). There is no way to obtain these automatically. This
client picks up from a URL/id+key a human already has (see
parse_recording_url()) and automates everything after that.

How a recording is fetched (confirmed by reading the source):
    1. GET  /api/recording/:id?key=...       -> recording metadata (used
       here just to fail fast with a clear error if the id/key are wrong)
    2. POST /api/recording/:id/cook?key=...  -> starts an async "cook"
       (transcode) job. Left at Craig's own defaults (format=flac,
       container=zip) deliberately — that's exactly the per-speaker FLAC
       zip structure transcribe_multitrack.py already expects, no new
       parsing needed.
    3. GET  /api/recording/:id/cook?key=...  -> poll job status:
       {ok, ready, download}. download.file is a filename once ready.
    4. GET  https://craig.chat/dl/<file>     -> the actual zip bytes.
       This is NOT under /api/ — a separate static-file route, confirmed
       from the page's own download-button handler.
"""

import asyncio
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp

BASE_URL = "https://craig.chat"
POLL_INTERVAL_SECONDS = 5
# Cooking a long multi-track recording into FLAC can take a while; generous timeout.
POLL_TIMEOUT_SECONDS = 900


class CraigError(Exception):
    """Any Craig API failure. Message is written to be safe/useful to show the user."""


def parse_recording_url(url_or_id: str, key: str | None = None) -> tuple:
    """
    Accept either a full Craig recording URL
    (https://craig.chat/rec/<id>?key=<key>, as Craig DMs/shows the user)
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


async def fetch_recording_info(session: aiohttp.ClientSession, rec_id: str, key: str) -> dict:
    """Fail fast with a clear error if the id/key are wrong, before triggering a cook job."""
    async with session.get(f"{BASE_URL}/api/recording/{rec_id}", params={"key": key}) as resp:
        try:
            data = await resp.json()
        except aiohttp.ContentTypeError:
            data = {"raw": await resp.text()}
        if resp.status != 200 or not data.get("ok"):
            raise CraigError(
                f"Craig rejected that recording ID/key (HTTP {resp.status}). This usually "
                f"means the link has expired (Craig keeps recordings 7 days) or was typed "
                f"wrong. Raw response: {data}"
            )
        return data.get("info", {})


async def start_cook(session: aiohttp.ClientSession, rec_id: str, key: str) -> None:
    async with session.post(
        f"{BASE_URL}/api/recording/{rec_id}/cook",
        params={"key": key},
        json={"format": "flac", "container": "zip", "dynaudnorm": False},
    ) as resp:
        try:
            data = await resp.json()
        except aiohttp.ContentTypeError:
            data = {"raw": await resp.text()}
        if resp.status != 200 or not data.get("ok"):
            raise CraigError(
                f"Craig refused to start cooking the recording (HTTP {resp.status}). "
                f"Raw response: {data}"
            )


async def wait_for_cook(session: aiohttp.ClientSession, rec_id: str, key: str) -> str:
    """Poll until the cook job is ready. Returns the filename to download."""
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        async with session.get(f"{BASE_URL}/api/recording/{rec_id}/cook", params={"key": key}) as resp:
            try:
                data = await resp.json()
            except aiohttp.ContentTypeError:
                data = {"raw": await resp.text()}
            if resp.status != 200 or not data.get("ok"):
                raise CraigError(
                    f"Craig reported an error while cooking (HTTP {resp.status}). "
                    f"Raw response: {data}"
                )
            if data.get("ready"):
                filename = (data.get("download") or {}).get("file")
                if not filename:
                    raise CraigError(f"Craig said the cook job is ready but gave no filename: {data}")
                return filename
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


async def fetch_and_extract_recording(url_or_id: str, dest_dir: Path, key: str | None = None) -> Path:
    """
    End-to-end: parse the recording URL, trigger cooking, wait for it,
    download, and extract into dest_dir (ready for transcribe_multitrack.py
    / pipeline.py). Returns dest_dir.
    """
    rec_id, rec_key = parse_recording_url(url_or_id, key)
    async with aiohttp.ClientSession() as session:
        await fetch_recording_info(session, rec_id, rec_key)
        await start_cook(session, rec_id, rec_key)
        filename = await wait_for_cook(session, rec_id, rec_key)
        zip_path = dest_dir.parent / f"_craig_{rec_id}.zip"
        await download_cooked_file(session, filename, zip_path)
        extract_craig_zip(zip_path, dest_dir)
        zip_path.unlink(missing_ok=True)
    return dest_dir
