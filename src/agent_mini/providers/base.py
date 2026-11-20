"""What every provider has in common: the types the loop works with, and the ``Provider`` interface.

Turning a server's raw tool calls into ``ToolCall`` objects is in ``parsing.py``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx


@dataclass
class ToolCall:
    """The model asking to run one tool. ``id`` links the tool's result back to this call."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """One reply from the model, the same shape for every provider: text, tool calls, or both."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    thinking: str | None = None
    usage: dict | None = None  # {prompt_tokens, completion_tokens, total_tokens}


@dataclass
class ModelInfo:
    """What the server reports about a model (only Ollama does today)."""

    params_b: float | None = None  # parameter count, in billions
    context_length: int | None = None
    capabilities: list[str] | None = None  # e.g. ["completion", "tools", "vision"]


# Called with each piece of text as it streams in.
StreamCallback = Callable[[str], Awaitable[None]]


def new_call_id() -> str:
    """Unique tool-call ID for servers that omit one (reusing call_0 confuses some)."""
    return f"call_{uuid.uuid4().hex[:12]}"


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text.strip()[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        err = err.get("message") or json.dumps(err)
    return str(err or data)[:300]


async def raise_for_status(
    resp: httpx.Response, hint: Callable[[int, str], str] | None = None
) -> None:
    """Like ``resp.raise_for_status()``, but the message carries the server's error text."""
    if resp.status_code < 400:
        return
    await resp.aread()
    message = _error_message(resp)
    extra = hint(resp.status_code, message) if hint else ""
    raise httpx.HTTPStatusError(
        f"HTTP {resp.status_code}: {message}" + (f" — {extra}" if extra else ""),
        request=resp.request,
        response=resp,
    )


class Provider:
    """An LLM endpoint. Subclasses turn its wire format into ``ChatResponse``."""

    name = "provider"
    # Context size the agent wants the server to allocate (Ollama num_ctx).
    context_window: int | None = None

    def __init__(self, model: str):
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        on_delta: StreamCallback | None = None,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        """One completion. With *on_delta*, text (and thinking) deltas are passed on as they arrive."""
        raise NotImplementedError

    async def model_info(self) -> ModelInfo | None:
        """Size, context length and capabilities as reported by the server, if it can tell."""
        return None

    async def close(self) -> None:
        """Release HTTP clients."""
