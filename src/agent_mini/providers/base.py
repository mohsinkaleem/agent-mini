"""Base provider interface and common types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool/function call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Standardised LLM response across all providers."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class BaseProvider(ABC):
    """Abstract base for LLM providers.

    Every provider must convert its native response into a ``ChatResponse``
    so the agent loop stays provider-agnostic.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send messages to the LLM and return a response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Currently configured model name."""
        ...
