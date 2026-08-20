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
    2. You paste the recording link Craig gives you into /meetingnotes,
       optionally attaching your own rough notes for that meeting too
       (falls back to MEETING_NOTES_FILE from .env if you don't).
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


# Discord's hard cap on a message's content is 2000 chars. Truncate anything
# dynamic (error messages, subprocess output) before sending — confirmed the hard
# way: an untruncated Craig API error response caused Discord itself to reject the
# message (400, "Must be 4000 or fewer in length" — that number is Discord's total
# request-body cap, not the per-field content limit, but the fix is the same:
# never send unbounded dynamic text as message content).
DISCORD_MESSAGE_LIMIT = 1900  # a bit under 2000 for safety margin


async def send_safe(channel, content: str, **kwargs) -> None:
    if len(content) > DISCORD_MESSAGE_LIMIT:
        content = content[: DISCORD_MESSAGE_LIMIT - 20] + "\n... (truncated)"
    await channel.send(content, **kwargs)


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
@app_commands.describe(
    url="The recording link Craig gave you (from /join's response or DM)",
    notes="Optional: your own rough notes for this meeting (.txt), used to ground the "
    "summary. Falls back to MEETING_NOTES_FILE from .env if you skip this.",
)
async def meetingnotes(interaction: discord.Interaction, url: str, notes: discord.Attachment | None = None):
    await interaction.response.send_message(
        f"Got it — processing that recording now. This can take a while (roughly "
        f"N × a few minutes, N = number of speakers, plus summarization). I'll post "
        f"the notes here when they're ready.",
    )
    channel = interaction.channel
    # Fire-and-forget: the interaction token isn't valid for the ~40+ min this can take,
    # so all further communication uses plain channel messages, not interaction followups.
    asyncio.create_task(process_recording(url, channel, notes))


async def process_recording(url: str, channel, notes_attachment=None) -> None:
    job_dir = BOT_WORK_DIR / f"job_{transcribe.make_timestamp()}"
    try:
        notes_file_override = None
        if notes_attachment is not None:
            job_dir.mkdir(parents=True, exist_ok=True)
            notes_file_override = job_dir / notes_attachment.filename
            await notes_attachment.save(notes_file_override)
            await channel.send(f"Using your attached notes ({notes_attachment.filename}) for this meeting.")

        await channel.send("Step 1/2: fetching and cooking the recording from Craig...")
        try:
            recording_dir = await craig_client.fetch_and_extract_recording(url, job_dir)
        except craig_client.CraigError as exc:
            await send_safe(channel, f"❌ Couldn't fetch the recording from Craig: {exc}")
            return

        await channel.send("Step 2/2: transcribing + summarizing (this is the slow part)...")
        stdout_text, returncode = await run_pipeline(recording_dir, notes_file_override)

        if returncode != 0:
            tail = "\n".join(stdout_text.splitlines()[-25:])
            await send_safe(
                channel, f"❌ Pipeline failed (exit code {returncode}). Last output:\n```\n{tail}\n```"
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
            await send_safe(channel, f"⚠️ Pipeline reported notes at `{notes_path}` but that file doesn't exist.")
            return

        await channel.send(
            content="✅ Done! Meeting notes:",
            file=discord.File(notes_path, filename=notes_path.name),
        )
        # Downloaded audio can be large (100+ MB across tracks) — clean it up now that
        # we've succeeded. Left in place on any failure path above, for debugging.
        cleanup_job_dir(job_dir)
    except Exception as exc:  # noqa: BLE001 - last resort: report, don't let the bot process die
        await send_safe(channel, f"❌ Unexpected error processing that recording: {exc}")


def cleanup_job_dir(job_dir: Path) -> None:
    import shutil

    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except OSError:
        pass  # best-effort; leftover disk usage isn't worth failing the job over


def participants_from_name_map(name_map: str | None) -> str | None:
    """
    Derive a --participants value from CRAIG_NAME_MAP's real-name side, so
    the summarization step gets both signals (relabeled transcript AND an
    explicit participant list) without a second config variable to keep in
    sync. E.g. "user1=Name1,user2=Name2" -> "Name1, Name2".
    """
    if not name_map:
        return None
    names = []
    for pair in name_map.split(","):
        if "=" in pair:
            names.append(pair.split("=", 1)[1].strip())
    return ", ".join(names) if names else None


async def run_pipeline(recording_dir: Path, notes_file_override: Path | None = None) -> tuple:
    """
    Run pipeline.py as a subprocess against recording_dir. Returns (stdout+stderr, returncode).

    notes_file_override: a per-meeting notes file (from /meetingnotes' optional
    attachment) takes priority over the static MEETING_NOTES_FILE from .env.
    """
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
        participants = participants_from_name_map(DEFAULT_NAME_MAP)
        if participants:
            args += ["--participants", participants]

    notes_file = str(notes_file_override) if notes_file_override else DEFAULT_NOTES_FILE
    if notes_file:
        args += ["--notes-file", notes_file]

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
