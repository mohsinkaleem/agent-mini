"""Base channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from ..providers.base import StreamCallback

# (channel_name, user_id, text, stream_callback?) → response text
MessageHandler = Callable[[str, str, str, StreamCallback | None], Awaitable[str]]


class BaseChannel(ABC):
    """Abstract base for chat-platform integrations."""

    @abstractmethod
    async def start(self, on_message: MessageHandler) -> None:
        """Start listening for incoming messages."""
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
