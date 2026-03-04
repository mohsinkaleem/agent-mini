"""Message bus — routes messages between chat channels and the agent."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .agent.loop import AgentLoop

log = logging.getLogger("agent-mini")

StreamEmitter = Callable[[str], Awaitable[None]]


class MessageBus:
    """Simple session-aware message router.

    Each ``(channel, user_id)`` pair gets its own conversation history so
    sessions stay isolated.
    """

    def __init__(self, agent: AgentLoop):
        self.agent = agent
        self.sessions: dict[str, list[dict]] = {}

    async def handle_message(
        self,
        channel: str,
        user_id: str,
        text: str,
        stream: StreamEmitter | None = None,
    ) -> str:
        """Route an incoming message through the agent and return the reply."""
        session_key = f"{channel}:{user_id}"
        session = self.sessions.setdefault(session_key, [])

        log.info("[%s:%s] → %s", channel, user_id, text[:120])
        response = await self.agent.run(text, session, on_stream=stream)
        log.info("[%s:%s] ← %s", channel, user_id, response[:120])

        return response

    def clear_session(self, channel: str, user_id: str) -> None:
        """Wipe conversation history for a specific user."""
        self.sessions.pop(f"{channel}:{user_id}", None)
