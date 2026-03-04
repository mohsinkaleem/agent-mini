"""Base channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

StreamEmitter = Callable[[str], Awaitable[None]]

# (channel_name, user_id, text, stream_emitter?) → response text
MessageHandler = Callable[[str, str, str, StreamEmitter | None], Awaitable[str]]


class BaseChannel(ABC):
    """Abstract base for chat-platform integrations."""

    @abstractmethod
    async def start(self, on_message: MessageHandler) -> None:
        """Start listening for incoming messages."""
        ...

    @abstractmethod
    async def send(self, user_id: str, text: str) -> None:
        """Proactively send a message to *user_id*."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier (e.g. ``telegram``)."""
        ...
