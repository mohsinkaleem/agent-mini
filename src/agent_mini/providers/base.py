"""Base provider interface and common types."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


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
    thinking: str | None = None


StreamCallback = Callable[[str], Awaitable[None]]


def parse_arguments(raw: Any) -> dict:
    """Normalize tool/function arguments into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_openai_tool_calls(raw: list[dict] | None) -> list[ToolCall] | None:
    """Convert OpenAI-style tool calls into ``ToolCall`` objects."""
    if not raw:
        return None

    calls: list[ToolCall] = []
    for i, tc in enumerate(raw):
        func = tc.get("function", {})
        calls.append(
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=func.get("name", ""),
                arguments=parse_arguments(func.get("arguments", {})),
            )
        )
    return calls or None


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

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        """Stream text deltas when supported; fallback to non-streaming."""
        response = await self.chat(messages, tools=tools, temperature=temperature)
        if response.thinking and on_thinking:
            await on_thinking(response.thinking)
        if response.content:
            await on_delta(response.content)
        return response

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
