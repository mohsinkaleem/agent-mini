"""Message bus — routes messages between chat channels and the agent."""

from __future__ import annotations

import asyncio
import logging

from .agent.loop import AgentLoop
from .providers.base import StreamCallback

log = logging.getLogger("agent-mini")


class MessageBus:
    """Simple session-aware message router.

    Each ``(channel, user_id)`` pair gets its own conversation history so
    sessions stay isolated. Messages from the same user run one at a time,
    so two quick messages can't edit the same history concurrently.
    """

    def __init__(self, agent: AgentLoop):
        self.agent = agent
        self.sessions: dict[str, list[dict]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def handle_message(
        self,
        channel: str,
        user_id: str,
        text: str,
        stream: StreamCallback | None = None,
    ) -> str:
        """Route an incoming message through the agent and return the reply."""
        session_key = f"{channel}:{user_id}"
        session = self.sessions.setdefault(session_key, [])
        lock = self._locks.setdefault(session_key, asyncio.Lock())

        log.info("[%s:%s] → %s", channel, user_id, text[:120])
        async with lock:
            response = await self.agent.run(text, session, on_stream=stream)
        log.info("[%s:%s] ← %s", channel, user_id, response[:120])

        return response
