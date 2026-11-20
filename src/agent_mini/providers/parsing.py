"""From what a model sends back to ``ToolCall`` objects.

The normal path is short: the server returns native tool calls whose arguments
are a JSON string (``parse_openai_tool_calls`` and ``parse_arguments``).

Below that are small-model fixes. Small models often send broken JSON, or write
the tool call into their reply text instead of making a real one.
``repair_json`` and ``parse_text_tool_calls`` recover what they meant. You can
skip them on a first read: with a large model they rarely do anything.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from .base import ToolCall, new_call_id

log = logging.getLogger("agent-mini")

# Set by parse_arguments when the raw arguments could not be turned into a dict,
# so the tool executor can tell the model instead of running with {}.
INVALID_ARGS_KEY = "__invalid_json__"


# ── The normal path ──────────────────────────────────────────────────


def parse_arguments(raw: Any) -> dict:
    """Turn a tool call's arguments (a dict, or a JSON string) into a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Small-model fix: try to repair the JSON before giving up.
        repaired = repair_json(raw)
        if repaired is not None:
            log.info("Repaired malformed tool arguments JSON")
            return repaired
        log.warning("Failed to parse tool arguments as JSON: %s", raw[:200])
        return {INVALID_ARGS_KEY: raw[:500]}
    if isinstance(parsed, dict):
        return parsed
    log.warning("Tool arguments parsed as %s instead of dict", type(parsed).__name__)
    return {INVALID_ARGS_KEY: raw[:500]}


def parse_openai_tool_calls(raw: list[dict] | None) -> list[ToolCall] | None:
    """Turn OpenAI-format calls (``{"id", "function": {"name", "arguments"}}``) into ``ToolCall`` objects."""
    calls = []
    for tc in raw or []:
        function = tc.get("function", {})
        calls.append(ToolCall(
            id=tc.get("id") or new_call_id(),
            name=function.get("name") or "",
            arguments=parse_arguments(function.get("arguments", {})),
        ))
    return calls or None


# ── Small-model fixes ────────────────────────────────────────────────


def repair_json(s: str) -> dict | None:
    """Fix the JSON mistakes small models make most often; None if it still doesn't parse."""
    # Markdown code fences around the JSON.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # Trailing commas: {"a": 1,} and [1, 2,]
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    # Single quotes instead of double quotes (only when there are no double quotes at all).
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    # Unquoted keys: {path: "a"} becomes {"path": "a"}
    s = re.sub(r"(?<=[{,\s])(\w+)\s*:", r'"\1":', s)
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# Ways models write tool calls as text when native tool calling fails:
# Hermes/Qwen tags, Mistral's marker, and fenced JSON blocks.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_MISTRAL_CALLS = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.DOTALL)
_FENCED_JSON = re.compile(r"```(?:json|tool_call|tool_code)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)

# A fenced JSON block with more than this much text around it is more likely
# an example in an answer than a tool call.
_EXAMPLE_TEXT_CHARS = 300


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return repair_json(text)


def _as_tool_calls(obj: Any, known: set[str]) -> list[ToolCall]:
    """Turn ``{"name", "arguments"|"parameters"}`` (or a list of them) into calls.

    Returns [] unless every item names a known tool.
    """
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
        for match in pattern.finditer(content):
            obj = _load_json(match.group(1))
            found = _as_tool_calls(obj, known) if obj is not None else []
            if found:
                calls.extend(found)
                rest = rest.replace(match.group(0), "", 1)
        looks_like_example = pattern is _FENCED_JSON and len(rest.strip()) > _EXAMPLE_TEXT_CHARS
        if calls and not looks_like_example:
            return calls, rest.strip()

    # The whole reply is one JSON object or list (Llama may prefix <|python_tag|>).
    bare = content.strip().removeprefix("<|python_tag|>").strip()
    if bare.startswith(("{", "[")):
        obj = _load_json(bare)
        calls = _as_tool_calls(obj, known) if obj is not None else []
        if calls:
            return calls, ""
    return [], content
