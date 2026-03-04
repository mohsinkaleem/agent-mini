"""Telegram channel — uses python-telegram-bot (v21+)."""

from __future__ import annotations

import logging

from .base import BaseChannel, MessageHandler

log = logging.getLogger("agent-mini")


class TelegramChannel(BaseChannel):
    """Connects to Telegram via the Bot API.

    Requires ``python-telegram-bot>=21``.  Install with::

        pip install agent-mini[telegram]
    """

    def __init__(self, token: str, allow_from: list[str] | None = None):
        self._token = token
        self._allow_from: set[str] | None = (
            set(allow_from) if allow_from else None
        )
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

    async def start(self, on_message: MessageHandler) -> None:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler as TGMsgHandler,
            filters,
        )

        self._handler = on_message
        self._app = Application.builder().token(self._token).build()

        # --- /start command ---
        async def _cmd_start(update: Update, ctx) -> None:  # noqa: ARG001
            await update.message.reply_text(
                "👋 Hi! I'm Agent Mini, your personal AI assistant.\n"
                "Send me any message to get started!"
            )

        # --- text messages ---
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

            # Typing indicator while the agent thinks
            await update.message.chat.send_action("typing")

            try:
                response = await self._handler("telegram", user_id, text)
                # Telegram message limit = 4096 chars
                for i in range(0, len(response), 4000):
                    chunk = response[i : i + 4000]
                    await update.message.reply_text(chunk, parse_mode=None)
            except Exception as e:
                log.error("[telegram] Error handling message: %s", e)
                await update.message.reply_text(f"⚠️ Error: {e}")

        self._app.add_handler(CommandHandler("start", _cmd_start))
        self._app.add_handler(
            TGMsgHandler(filters.TEXT & ~filters.COMMAND, _on_text)
        )

        log.info("[telegram] Starting bot…")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def send(self, user_id: str, text: str) -> None:
        if self._app:
            for i in range(0, len(text), 4000):
                await self._app.bot.send_message(
                    chat_id=int(user_id), text=text[i : i + 4000]
                )

    async def stop(self) -> None:
        if self._app:
            log.info("[telegram] Stopping…")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
