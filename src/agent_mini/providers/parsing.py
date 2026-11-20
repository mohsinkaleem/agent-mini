"""From what a model sends back to ``ToolCall`` objects.

The normal path is short: the server returns native tool calls whose arguments
are a JSON string (``parse_openai_tool_calls`` and ``parse_arguments``).

Below that is a small-model fix: small models sometimes send broken JSON, and
``repair_json`` recovers what they meant. You can skip it on a first read: with
a large model it rarely does anything.
"""

from __future__ import annotations

import json
import logging
import re
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


# ── Small-model fix ─────────────────────────────────────────────────


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
