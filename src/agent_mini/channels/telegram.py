"""Telegram channel — uses python-telegram-bot (v21+)."""

from __future__ import annotations

import asyncio
import logging
import time

from .base import BaseChannel, MessageHandler

log = logging.getLogger("agent-mini")


class TelegramChannel(BaseChannel):
    """Connects to Telegram via the Bot API.

    Requires ``python-telegram-bot>=21``. Install with::

        pip install agent-mini[telegram]
    """

    def __init__(
        self,
        token: str,
        allow_from: list[str] | None = None,
        stream_responses: bool = True,
    ):
        self._token = token
        self._allow_from: set[str] | None = set(allow_from) if allow_from else None
        self._stream_responses = stream_responses
        self._app = None
        self._handler: MessageHandler | None = None

    @property
    def name(self) -> str:
        return "telegram"

    def _is_allowed(self, user_id: str, username: str | None = None) -> bool:
        if not self._allow_from or "*" in self._allow_from:
            return True
        if user_id in self._allow_from:
            return True
        if username and username in self._allow_from:
            return True
        return False

    @staticmethod
    def _chunk_text(text: str, size: int = 4000) -> list[str]:
        if not text:
            return ["(empty response)"]
        return [text[i : i + size] for i in range(0, len(text), size)]

    async def start(self, on_message: MessageHandler) -> None:
        from telegram import Update
        from telegram.error import BadRequest
        from telegram.ext import (
            Application,
            CommandHandler,
            filters,
        )
        from telegram.ext import (
            MessageHandler as TGMsgHandler,
        )

        self._handler = on_message
        self._app = Application.builder().token(self._token).build()

        async def _cmd_start(update: Update, ctx) -> None:  # noqa: ARG001
            await update.message.reply_text(
                "Hi, I'm Agent Mini. Send me a message to begin."
            )

        async def _on_text(update: Update, ctx) -> None:  # noqa: ARG001
            if not update.message or not update.message.text:
                return

            user = update.message.from_user
            user_id = str(user.id)
            username = user.username or ""

            if not self._is_allowed(user_id, username):
                log.debug("[telegram] Ignoring disallowed user %s", user_id)
                return

            text = update.message.text
            log.info("[telegram] %s: %s", username or user_id, text[:120])
            await update.message.chat.send_action("typing")

            placeholder = None
            try:
                if self._stream_responses:
                    placeholder = await update.message.reply_text("...")
                    buffer: list[str] = []
                    lock = asyncio.Lock()
                    last_state = {"text": "...", "t": 0.0}

                    async def _flush(force: bool = False) -> None:
                        preview = "".join(buffer)[:4000] or "..."
                        now = time.monotonic()
                        if (
                            not force
                            and preview == last_state["text"]
                            or (
                                not force
                                and now - last_state["t"] < 0.7
                                and len(preview) - len(last_state["text"]) < 80
                            )
                        ):
                            return
                        async with lock:
                            if preview == last_state["text"]:
                                return
                            try:
                                await placeholder.edit_text(preview, parse_mode=None)
                                last_state["text"] = preview
                                last_state["t"] = now
                            except BadRequest as e:
                                if "message is not modified" not in str(e).lower():
                                    raise

                    async def _emit(delta: str) -> None:
                        if not delta:
                            return
                        buffer.append(delta)
                        await _flush(force=False)

                    response = await self._handler("telegram", user_id, text, _emit)
                    await _flush(force=True)

                    chunks = self._chunk_text(response)
                    if chunks[0] != last_state["text"]:
                        await placeholder.edit_text(chunks[0], parse_mode=None)
                    for chunk in chunks[1:]:
                        await update.message.reply_text(chunk, parse_mode=None)
                else:
                    response = await self._handler("telegram", user_id, text, None)
                    for chunk in self._chunk_text(response):
                        await update.message.reply_text(chunk, parse_mode=None)
            except Exception as e:
                log.error("[telegram] Error handling message: %s", e)
                if placeholder is not None:
                    await placeholder.edit_text(f"Error: {e}", parse_mode=None)
                else:
                    await update.message.reply_text(f"Error: {e}", parse_mode=None)

        self._app.add_handler(CommandHandler("start", _cmd_start))
        self._app.add_handler(TGMsgHandler(filters.TEXT & ~filters.COMMAND, _on_text))

        log.info("[telegram] Starting bot...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def send(self, user_id: str, text: str) -> None:
        if self._app:
            for chunk in self._chunk_text(text):
                await self._app.bot.send_message(chat_id=int(user_id), text=chunk)

    async def stop(self) -> None:
        if self._app:
            log.info("[telegram] Stopping...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
