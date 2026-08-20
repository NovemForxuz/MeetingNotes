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

craig_client.py's API calls have been verified against a real live
recording (see its docstring/README). The bot's own Discord-facing flow
(slash command, progress bar, file attachments) is still worth a live
`/meetingnotes` run to confirm end-to-end.
"""

import asyncio
import os
import re
import sys
import time
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

# Recognized lines from transcribe_multitrack.py / summarize.py's own stdout,
# used to derive a live progress fraction for the combined "transcribe +
# summarize" stage. Weighted 90/10 transcription/summarization, matching real
# observed timing (e.g. a real ~31 min, 5-track meeting: ~38 min transcribing
# vs ~1-2 min summarizing) — transcription dominates the wall-clock time.
TRACK_PROGRESS_RE = re.compile(r"Transcribing track (\d+)/(\d+)")
PASS_PROGRESS_RE = re.compile(r"Pass (\d+)/3")


def parse_pipeline_progress(line: str) -> tuple | None:
    """Return (fraction, label) if line is a recognized progress marker, else None."""
    match = TRACK_PROGRESS_RE.search(line)
    if match:
        done, total = int(match.group(1)), int(match.group(2))
        return 0.9 * (done / total), f"transcribing track {done}/{total}"

    match = PASS_PROGRESS_RE.search(line)
    if match:
        current = int(match.group(1))
        return 0.9 + 0.1 * (current / 3), f"summarizing (pass {current}/3)"

    if "Loading Whisper model" in line:
        return 0.02, "loading Whisper model"
    if line.startswith("Found ") and "track" in line:
        return 0.03, "starting transcription"
    if "Transcribing '" in line:  # single-file (non-multitrack) path
        return 0.1, "transcribing"
    if "Generating structured meeting notes" in line:
        return 0.9, "summarizing"
    return None

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Slash commands registered globally (the default) can take up to an hour for
# Discord to propagate to clients — confirmed the hard way: a newly-added command
# parameter didn't show up in the Discord UI right after a bot restart. Setting
# DISCORD_GUILD_ID scopes the command to one server instead, which syncs
# instantly — worth doing for a single-server bot like this one regardless of
# the propagation-delay issue, since it also makes future command changes
# immediately testable.
_raw_guild_id = os.environ.get("DISCORD_GUILD_ID")
try:
    GUILD_OBJECT = discord.Object(id=int(_raw_guild_id)) if _raw_guild_id else None
except ValueError:
    print(
        f"WARNING: DISCORD_GUILD_ID='{_raw_guild_id}' is not a valid integer — "
        f"ignoring it and falling back to a slow global command sync.",
        file=sys.stderr,
    )
    GUILD_OBJECT = None


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


def format_bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return "`[" + "#" * filled + "-" * (width - filled) + f"]` {fraction * 100:.0f}%"


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ProgressReporter:
    """
    Wraps a single Discord message, edited in place, to show a live
    stage/progress-bar/elapsed-time indicator instead of a wall of separate
    status messages. Auto-refreshes on a timer too, so elapsed time keeps
    moving even between real progress events (e.g. while Whisper is loading
    a model, with no line of output for a while).

    Edits are throttled (MIN_EDIT_INTERVAL) to stay well clear of Discord's
    per-channel/per-message rate limits regardless of how often update() is
    called from real progress events.
    """

    MIN_EDIT_INTERVAL = 4.0  # seconds

    def __init__(self, message: discord.Message, refresh_seconds: float = 12.0):
        self.message = message
        self.start_time = time.monotonic()
        self.stage = "Starting..."
        self.fraction = 0.0
        self.label = ""
        self._last_edit = 0.0
        self._refresh_seconds = refresh_seconds
        self._refresh_task: asyncio.Task | None = None

    def start_auto_refresh(self) -> None:
        self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

    def stop_auto_refresh(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def _auto_refresh_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refresh_seconds)
                await self._render(force=True)
        except asyncio.CancelledError:
            pass

    def render_text(self) -> str:
        elapsed = format_elapsed(time.monotonic() - self.start_time)
        bar = format_bar(self.fraction)
        suffix = f" — {self.label}" if self.label else ""
        return f"**{self.stage}**\n{bar}{suffix}\nElapsed: {elapsed}"

    async def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_edit) < self.MIN_EDIT_INTERVAL:
            return
        self._last_edit = now
        try:
            await self.message.edit(content=self.render_text())
        except discord.HTTPException:
            pass  # a failed/rate-limited edit shouldn't take down the whole job

    async def update(self, stage: str | None = None, fraction: float | None = None, label: str | None = None) -> None:
        if stage is not None:
            self.stage = stage
        if fraction is not None:
            self.fraction = fraction
        if label is not None:
            self.label = label
        await self._render()

    async def finish(self, final_text: str) -> None:
        self.stop_auto_refresh()
        try:
            await self.message.edit(content=final_text)
        except discord.HTTPException:
            pass


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


@tree.command(name="meetingnotes", description="Turn a Craig recording link into meeting notes", guild=GUILD_OBJECT)
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
    overall_start = time.monotonic()
    progress_message = await channel.send("**Starting...**\n" + format_bar(0.0) + "\nElapsed: 0s")
    reporter = ProgressReporter(progress_message)
    reporter.start_auto_refresh()
    try:
        notes_file_override = None
        if notes_attachment is not None:
            job_dir.mkdir(parents=True, exist_ok=True)
            notes_file_override = job_dir / notes_attachment.filename
            await notes_attachment.save(notes_file_override)
            await channel.send(f"Using your attached notes ({notes_attachment.filename}) for this meeting.")

        await reporter.update(stage="Step 1/2: Fetching & cooking from Craig", fraction=0.0)
        try:
            recording_dir = await craig_client.fetch_and_extract_recording(
                url, job_dir, on_progress=lambda frac, label: reporter.update(fraction=frac, label=label)
            )
        except craig_client.CraigError as exc:
            await reporter.finish(f"❌ Failed at Step 1/2 (fetching from Craig) after {format_elapsed(time.monotonic() - overall_start)}.")
            await send_safe(channel, f"❌ Couldn't fetch the recording from Craig: {exc}")
            return

        await reporter.update(stage="Step 2/2: Transcribing + summarizing", fraction=0.0, label="starting")

        async def on_line(line: str) -> None:
            parsed = parse_pipeline_progress(line)
            if parsed:
                await reporter.update(fraction=parsed[0], label=parsed[1])

        stdout_text, returncode = await run_pipeline(recording_dir, notes_file_override, on_line=on_line)

        if returncode != 0:
            tail = "\n".join(stdout_text.splitlines()[-25:])
            await reporter.finish(f"❌ Failed at Step 2/2 after {format_elapsed(time.monotonic() - overall_start)}.")
            await send_safe(
                channel, f"❌ Pipeline failed (exit code {returncode}). Last output:\n```\n{tail}\n```"
            )
            return

        notes_match = SAVED_NOTES_RE.search(stdout_text)
        if not notes_match:
            await reporter.finish(f"⚠️ Finished after {format_elapsed(time.monotonic() - overall_start)}, but something's off.")
            await channel.send(
                "⚠️ Pipeline finished but I couldn't find the notes file path in its output. "
                "Check the output/ folder directly."
            )
            return

        notes_path = Path(notes_match.group(1).strip())
        if not notes_path.exists():
            await reporter.finish(f"⚠️ Finished after {format_elapsed(time.monotonic() - overall_start)}, but something's off.")
            await send_safe(channel, f"⚠️ Pipeline reported notes at `{notes_path}` but that file doesn't exist.")
            return

        total_elapsed = format_elapsed(time.monotonic() - overall_start)
        await reporter.finish(f"✅ **Done in {total_elapsed}!**")
        await channel.send(
            content="Meeting notes:",
            file=discord.File(notes_path, filename=notes_path.name),
        )
        # Downloaded audio can be large (100+ MB across tracks) — clean it up now that
        # we've succeeded. Left in place on any failure path above, for debugging.
        cleanup_job_dir(job_dir)
    except Exception as exc:  # noqa: BLE001 - last resort: report, don't let the bot process die
        await reporter.finish(f"❌ Unexpected error after {format_elapsed(time.monotonic() - overall_start)}.")
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


async def run_pipeline(recording_dir: Path, notes_file_override: Path | None = None, on_line=None) -> tuple:
    """
    Run pipeline.py as a subprocess against recording_dir. Returns (stdout+stderr, returncode).

    notes_file_override: a per-meeting notes file (from /meetingnotes' optional
    attachment) takes priority over the static MEETING_NOTES_FILE from .env.

    on_line, if given, is awaited with each line of output as it's produced
    (not just after the whole subprocess finishes), so callers can drive a
    live progress indicator — see parse_pipeline_progress().
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

    lines = []
    while True:
        raw_line = await process.stdout.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        lines.append(line)
        if on_line:
            await on_line(line)

    returncode = await process.wait()
    return "\n".join(lines), returncode


@client.event
async def on_ready():
    if GUILD_OBJECT is not None:
        # Clean up any stale GLOBAL registration from before DISCORD_GUILD_ID was set —
        # otherwise Discord may show a duplicate, outdated /meetingnotes for a while.
        tree.clear_commands(guild=None)
        await tree.sync()
        await tree.sync(guild=GUILD_OBJECT)
        print(f"Logged in as {client.user}. Slash command /meetingnotes synced to guild {GUILD_OBJECT.id} (instant).")
    else:
        await tree.sync()
        print(
            f"Logged in as {client.user}. Slash command /meetingnotes synced GLOBALLY — "
            f"this can take up to an hour to show up in Discord. Set DISCORD_GUILD_ID in "
            f".env for instant sync to one server instead."
        )


def main() -> None:
    check_config()
    BOT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    client.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
