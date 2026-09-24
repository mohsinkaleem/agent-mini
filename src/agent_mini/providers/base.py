"""Base provider interface and common types."""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("agent-mini")


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
    usage: dict | None = None  # {prompt_tokens, completion_tokens, total_tokens}


@dataclass
class ModelInfo:
    """What the server reports about a model (only Ollama does today)."""

    params_b: float | None = None  # parameter count, in billions
    context_length: int | None = None
    capabilities: list[str] | None = None  # e.g. ["completion", "tools", "vision"]


StreamCallback = Callable[[str], Awaitable[None]]

# Set by parse_arguments when the raw arguments could not be turned into a dict,
# so the tool executor can tell the model instead of running with {}.
INVALID_ARGS_KEY = "__invalid_json__"


def new_call_id() -> str:
    """Unique tool-call ID for servers that omit one (reusing call_0 confuses some)."""
    return f"call_{uuid.uuid4().hex[:12]}"


def _repair_json(s: str) -> dict | None:
    """Attempt common JSON repairs for malformed small-model outputs."""
    # Strip markdown code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # Fix trailing commas
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    # Fix single quotes → double quotes (simple heuristic)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    # Fix unquoted keys: word: -> "word":
    s = re.sub(r"(?<=[{,\s])(\w+)\s*:", r'"\1":', s)
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


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
            if isinstance(parsed, dict):
                return parsed
            log.warning("Tool arguments parsed as %s instead of dict", type(parsed).__name__)
            return {INVALID_ARGS_KEY: raw[:500]}
        except json.JSONDecodeError:
            # Attempt repair before giving up
            repaired = _repair_json(raw)
            if repaired is not None:
                log.info("Repaired malformed tool arguments JSON")
                return repaired
            log.warning("Failed to parse tool arguments as JSON: %s", raw[:200])
            return {INVALID_ARGS_KEY: raw[:500]}
    return {}


def parse_openai_tool_calls(raw: list[dict] | None) -> list[ToolCall] | None:
    """Convert OpenAI-style tool calls into ``ToolCall`` objects."""
    if not raw:
        return None

    calls: list[ToolCall] = []
    for tc in raw:
        func = tc.get("function", {})
        calls.append(
            ToolCall(
                id=tc.get("id") or new_call_id(),
                name=func.get("name", ""),
                arguments=parse_arguments(func.get("arguments", {})),
            )
        )
    return calls or None


# Ways models write tool calls as text when native tool calling fails:
# Hermes/Qwen tags, Mistral's marker, and fenced JSON blocks.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_MISTRAL_CALLS = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.DOTALL)
_FENCED_JSON = re.compile(r"```(?:json|tool_call|tool_code)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _repair_json(text)


def _as_tool_calls(obj: Any, known: set[str]) -> list[ToolCall]:
    """Turn ``{"name", "arguments"|"parameters"}`` (or a list of them) into calls."""
    calls: list[ToolCall] = []
    for item in obj if isinstance(obj, list) else [obj]:
        if not isinstance(item, dict):
            return []
        if isinstance(item.get("function"), dict):
            item = item["function"]
        if item.get("name") not in known:
            return []
        args = item.get("arguments", item.get("parameters", {}))
        calls.append(ToolCall(id=new_call_id(), name=item["name"], arguments=parse_arguments(args)))
    return calls


def parse_text_tool_calls(content: str | None, known: Iterable[str]) -> tuple[list[ToolCall], str]:
    """Recover tool calls a model wrote in its text instead of as native tool calls.

    Only calls to *known* tool names count. Returns ``(calls, remaining_text)``;
    ``calls`` is empty when nothing was found.
    """
    known = set(known)
    if not content or not any(name in content for name in known):
        return [], content or ""
    for pattern in (_TOOL_CALL_TAG, _MISTRAL_CALLS, _FENCED_JSON):
        calls: list[ToolCall] = []
        rest = content
        for m in pattern.finditer(content):
            obj = _load_json(m.group(1))
            found = _as_tool_calls(obj, known) if obj is not None else []
            if found:
                calls.extend(found)
                rest = rest.replace(m.group(0), "", 1)
        # A fenced block inside a long answer is more likely an example than a call.
        if calls and not (pattern is _FENCED_JSON and len(rest.strip()) > 300):
            return calls, rest.strip()
    bare = content.strip().removeprefix("<|python_tag|>").strip()
    if bare.startswith(("{", "[")):
        obj = _load_json(bare)
        calls = _as_tool_calls(obj, known) if obj is not None else []
        if calls:
            return calls, ""
    return [], content


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


class BaseProvider(ABC):
    """Abstract base for LLM providers.

    Every provider must convert its native response into a ``ChatResponse``
    so the agent loop stays provider-agnostic.
    """

    # Context size the agent wants the server to allocate (Ollama num_ctx).
    context_window: int | None = None

    async def model_info(self) -> ModelInfo | None:
        """Size, context length and capabilities as reported by the server, if supported."""
        return None

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
    ) -> ChatResponse:
        """Send messages to the LLM and return a response."""
        ...

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        """Stream text deltas when supported; fallback to non-streaming."""
        response = await self.chat(messages, tools=tools, temperature=temperature)
        if response.thinking and on_thinking:
            await on_thinking(response.thinking)
        if response.content:
            await on_delta(response.content)
        return response

    async def close(self) -> None:
        """Clean up resources (e.g. HTTP clients). Override if needed."""

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
