#!/usr/bin/env python3
"""
discord_bot.py — Discord bot that turns a pasted Craig recording link into
posted-back meeting notes, automating everything after the one unavoidable
manual step.

Why there's a manual step at all: Craig deliberately keeps a recording's
id+key private (DMed to whoever ran /join, or shown only to them as a
fallback) — there is no way for a bot to obtain this automatically (see
craig_client.py's docstring for how this was confirmed). So the real-world
flow is:

    1. You run /join and /stop in Discord, same as ever.
    2. You paste the recording link Craig gives you into /meetingnotes.
    3. Everything else is automatic: cook the recording into per-speaker
       FLAC via Craig's own API, download, extract, run the full
       transcription + three-pass summarization pipeline, and post the
       resulting notes back into the channel.

Deliberately shells out to pipeline.py as a SEPARATE PROCESS rather than
importing/calling its functions in-process. transcribe_multitrack.py and
summarize.py call sys.exit() on errors by design — correct for a one-shot
CLI script, but importing them here would mean a single failed meeting
could kill this long-running bot for every future meeting too. A
subprocess crashing is just a failed job; this process stays up.

Setup:
    1. Create a Discord application + bot user at
       https://discord.com/developers/applications, enable it, invite it
       to your server (needs: View Channel, Send Messages, Attach Files,
       Use Application Commands in the channel you'll use it from).
    2. Put the bot token in transcription/.env as DISCORD_BOT_TOKEN=...
       (never commit this — .env is gitignored).
    3. Optionally set CRAIG_NAME_MAP and MEETING_NOTES_FILE in .env so you
       don't have to pass them every time (see .env.example).
    4. pip install -r requirements.txt (adds discord.py, aiohttp)
    5. python discord_bot.py — leave it running during/after your meeting.

This has not been tested against a real live Craig recording yet. Treat
the first real run as a live test.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import discord
from discord import app_commands

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import craig_client
import transcribe

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DEFAULT_NAME_MAP = os.environ.get("CRAIG_NAME_MAP")  # e.g. "user1=Name1,user2=Name2"
DEFAULT_NOTES_FILE = os.environ.get("MEETING_NOTES_FILE")  # optional path to rough notes
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")

PIPELINE_SCRIPT = Path(__file__).parent / "pipeline.py"
BOT_WORK_DIR = Path(__file__).parent / "input" / "_bot_jobs"

SAVED_NOTES_RE = re.compile(r"Saved meeting notes to: (.+)")
SAVED_TRANSCRIPT_RE = re.compile(r"Saved (?:merged transcript|transcript) to: (.+)")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


def check_config() -> None:
    if not BOT_TOKEN:
        print(
            "ERROR: DISCORD_BOT_TOKEN is not set.\n"
            "Set it as an environment variable, or add it to transcription/.env:\n"
            "    DISCORD_BOT_TOKEN=your-bot-token-here\n\n"
            "Create a bot at https://discord.com/developers/applications.",
            file=sys.stderr,
        )
        sys.exit(1)


@tree.command(name="meetingnotes", description="Turn a Craig recording link into meeting notes")
@app_commands.describe(url="The recording link Craig gave you (from /join's response or DM)")
async def meetingnotes(interaction: discord.Interaction, url: str):
    await interaction.response.send_message(
        f"Got it — processing that recording now. This can take a while (roughly "
        f"N × a few minutes, N = number of speakers, plus summarization). I'll post "
        f"the notes here when they're ready.",
    )
    channel = interaction.channel
    # Fire-and-forget: the interaction token isn't valid for the ~40+ min this can take,
    # so all further communication uses plain channel messages, not interaction followups.
    asyncio.create_task(process_recording(url, channel))


async def process_recording(url: str, channel) -> None:
    job_dir = BOT_WORK_DIR / f"job_{transcribe.make_timestamp()}"
    try:
        await channel.send("Step 1/2: fetching and cooking the recording from Craig...")
        try:
            recording_dir = await craig_client.fetch_and_extract_recording(url, job_dir)
        except craig_client.CraigError as exc:
            await channel.send(f"❌ Couldn't fetch the recording from Craig: {exc}")
            return

        await channel.send("Step 2/2: transcribing + summarizing (this is the slow part)...")
        stdout_text, returncode = await run_pipeline(recording_dir)

        if returncode != 0:
            tail = "\n".join(stdout_text.splitlines()[-25:])
            await channel.send(
                f"❌ Pipeline failed (exit code {returncode}). Last output:\n```\n{tail}\n```"
            )
            return

        notes_match = SAVED_NOTES_RE.search(stdout_text)
        if not notes_match:
            await channel.send(
                "⚠️ Pipeline finished but I couldn't find the notes file path in its output. "
                "Check the output/ folder directly."
            )
            return

        notes_path = Path(notes_match.group(1).strip())
        if not notes_path.exists():
            await channel.send(f"⚠️ Pipeline reported notes at `{notes_path}` but that file doesn't exist.")
            return

        await channel.send(
            content="✅ Done! Meeting notes:",
            file=discord.File(notes_path, filename=notes_path.name),
        )
        # Downloaded audio can be large (100+ MB across tracks) — clean it up now that
        # we've succeeded. Left in place on any failure path above, for debugging.
        cleanup_job_dir(job_dir)
    except Exception as exc:  # noqa: BLE001 - last resort: report, don't let the bot process die
        await channel.send(f"❌ Unexpected error processing that recording: {exc}")


def cleanup_job_dir(job_dir: Path) -> None:
    import shutil

    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except OSError:
        pass  # best-effort; leftover disk usage isn't worth failing the job over


async def run_pipeline(recording_dir: Path) -> tuple:
    """Run pipeline.py as a subprocess against recording_dir. Returns (stdout+stderr, returncode)."""
    args = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        str(recording_dir),
        "--model",
        DEFAULT_WHISPER_MODEL,
        "--language",
        DEFAULT_LANGUAGE,
    ]
    if DEFAULT_NAME_MAP:
        args += ["--name-map", DEFAULT_NAME_MAP]
    if DEFAULT_NOTES_FILE:
        args += ["--notes-file", DEFAULT_NOTES_FILE]

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
    )
    stdout_bytes, _ = await process.communicate()
    return stdout_bytes.decode("utf-8", errors="replace"), process.returncode


@client.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {client.user}. Slash command /meetingnotes is ready.")


def main() -> None:
    check_config()
    BOT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    client.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
