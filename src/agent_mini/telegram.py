"""Telegram gateway: one conversation per user, replies streamed into the chat.

Needs ``python-telegram-bot>=21`` (``pip install agent-mini[telegram]``).
"""

from __future__ import annotations

import asyncio
import logging
import time

from .loop import AgentLoop
from .providers.base import StreamCallback

log = logging.getLogger("agent-mini")


def gateway_config(config: dict) -> tuple[dict, list[str]]:
    """The config the gateway should run with, and warnings to show at startup.

    A public bot (empty ``allowFrom`` or ``"*"``) runs read-only unless
    ``allowShell`` is set. With more than one possible user, memory is off
    unless ``sharedMemory`` is set, so one user's facts don't reach another's prompt.
    """
    tg = config["channels"]["telegram"]
    allow = [str(u) for u in tg["allowFrom"]]
    warnings = []
    public = not allow or "*" in allow
    if public and tg["allowShell"]:
        warnings.append("Telegram bot is PUBLIC and the shell is enabled: anyone who finds the bot "
                        "can run commands on this machine. Set channels.telegram.allowFrom to your numeric user ID.")
    elif public:
        config = {**config, "tools": {**config["tools"], "sandboxLevel": "readonly"}}
        warnings.append("Telegram bot is PUBLIC: anyone who finds it can use it. Running read-only "
                        "(no shell, no file writes). Set channels.telegram.allowFrom to your numeric user ID, "
                        "or channels.telegram.allowShell: true to accept the risk.")
    elif any(not u.isdigit() for u in allow):
        warnings.append("channels.telegram.allowFrom contains usernames. Usernames can be changed and then "
                        "claimed by someone else; prefer numeric user IDs.")
    if (public or len(allow) > 1) and not tg["sharedMemory"]:
        config = {**config, "memory": {**config["memory"], "enabled": False}}
        warnings.append("Memory is off because several users can reach the bot "
                        "(channels.telegram.sharedMemory: true shares one store).")
    return config, warnings


def chunk_text(text: str, size: int = 4000) -> list[str]:
    """Telegram messages are capped at 4096 characters."""
    return [text[i : i + size] for i in range(0, len(text), size)] or ["(empty response)"]


class TelegramGateway:
    """Routes each Telegram user to their own conversation with *agent*, one message at a time."""

    def __init__(self, agent: AgentLoop, token: str, allow_from: list[str], stream: bool = True):
        self.agent = agent
        self._token = token
        self._allow = {str(u) for u in allow_from}
        self._stream = stream
        self._app = None
        self.sessions: dict[str, list[dict]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._seen: set[str] = set()

    @property
    def is_public(self) -> bool:
        return not self._allow or "*" in self._allow

    def is_allowed(self, user_id: str, username: str = "") -> bool:
        return self.is_public or user_id in self._allow or bool(username and username in self._allow)

    async def handle(self, user_id: str, text: str, stream: StreamCallback | None = None) -> str:
        """Run one message through the agent. Messages from the same user wait their turn."""
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            return await self.agent.run(text, self.sessions.setdefault(user_id, []), on_stream=stream)

    async def start(self) -> None:
        from telegram import Update
        from telegram.error import BadRequest
        from telegram.ext import Application, CommandHandler, MessageHandler, filters

        async def on_start(update: Update, _ctx) -> None:
            await update.message.reply_text("Hi, I'm Agent Mini. Send me a message to begin.")

        async def on_reset(update: Update, _ctx) -> None:
            self.sessions.pop(str(update.message.from_user.id), None)
            await update.message.reply_text("Conversation cleared.")

        async def on_text(update: Update, _ctx) -> None:
            message = update.message
            if not message or not message.text:
                return
            user_id, username = str(message.from_user.id), message.from_user.username or ""
            if not self.is_allowed(user_id, username):
                log.debug("[telegram] ignoring user %s", user_id)
                return
            if self.is_public and user_id not in self._seen:
                self._seen.add(user_id)
                log.warning("[telegram] new user on a public bot: id=%s username=%s", user_id, username or "-")
            log.info("[telegram] %s: %s", username or user_id, message.text[:120])
            await message.chat.send_action("typing")
            placeholder = await message.reply_text("...") if self._stream else None
            shown = {"text": "...", "t": 0.0}

            async def show(text: str, force: bool = False) -> None:
                # Telegram rate-limits edits: at most one every 0.7 s, plus a final one.
                now = time.monotonic()
                text = text[:4000] or "..."
                if text == shown["text"] or (not force and now - shown["t"] < 0.7):
                    return
                shown.update(text=text, t=now)
                try:
                    await placeholder.edit_text(text, parse_mode=None)
                except BadRequest as e:
                    if "message is not modified" not in str(e).lower():
                        raise

            buffer: list[str] = []

            async def on_delta(delta: str) -> None:
                buffer.append(delta)
                await show("".join(buffer))

            try:
                reply = await self.handle(user_id, message.text, on_delta if placeholder else None)
                chunks = chunk_text(reply)
                if placeholder:
                    await show(chunks.pop(0), force=True)
                for chunk in chunks:
                    await message.reply_text(chunk, parse_mode=None)
            except Exception as e:
                log.error("[telegram] error handling message: %s", e)
                if placeholder:
                    await placeholder.edit_text(f"Error: {e}", parse_mode=None)
                else:
                    await message.reply_text(f"Error: {e}", parse_mode=None)

        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(CommandHandler("start", on_start))
        self._app.add_handler(CommandHandler("reset", on_reset))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
