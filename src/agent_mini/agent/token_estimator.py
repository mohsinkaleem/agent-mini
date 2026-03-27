"""Lightweight token estimation and model tier classification."""

from __future__ import annotations

import json
import re


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a message list."""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead (role, delimiters)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Vision messages: list of text/image parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict):
                    total += 85  # image token estimate
        # Tool calls in assistant messages
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total += estimate_tokens(func.get("name", ""))
            args = func.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args)
            total += estimate_tokens(args)
    return total


# ── Model tier classification ────────────────────────────────────────

_TIER_PATTERNS: list[tuple[str, str]] = [
    # Cloud / API models
    (r"gemini|gpt-4|gpt-3\.5|claude|deepseek-v[23]", "cloud"),
    # Medium (9-14B)
    (r"14b|12b|13b|nemo", "medium"),
    # Small (4-8B)
    (r"[78]b|:7b|:8b|mistral(?!.*nemo)", "small"),
    # Tiny (1-3B)
    (r"[123]\.?\d*b|:1b|:3b|phi-4-mini|gemma3?:1b", "tiny"),
]


def classify_model_tier(model_name: str) -> str:
    """Classify a model name into tiny / small / medium / cloud."""
    name = model_name.lower()
    for pattern, tier in _TIER_PATTERNS:
        if re.search(pattern, name):
            return tier
    return "small"  # safe default


# Effective context: the token count where accuracy stays high (~90%).
_EFFECTIVE_CONTEXT = {
    "tiny": 3000,
    "small": 6000,
    "medium": 12000,
    "cloud": 32000,
}


def get_effective_context(model_name: str) -> int:
    """Return the effective (usable) context budget for a model."""
    tier = classify_model_tier(model_name)
    return _EFFECTIVE_CONTEXT.get(tier, 6000)


# Default max iterations per tier (used when user hasn't set a custom value).
_TIER_MAX_ITERATIONS = {
    "tiny": 10,
    "small": 15,
    "medium": 20,
    "cloud": 25,
}


def get_tier_max_iterations(model_name: str) -> int:
    """Suggested max iterations for a model tier."""
    tier = classify_model_tier(model_name)
    return _TIER_MAX_ITERATIONS.get(tier, 20)


# Tool output size limits per tier.
_TIER_OUTPUT_LIMITS = {
    "tiny": 2000,
    "small": 4000,
    "medium": 8000,
    "cloud": 50000,
}


def get_output_limit(model_name: str) -> int:
    """Max chars for a single tool output, based on model tier."""
    tier = classify_model_tier(model_name)
    return _TIER_OUTPUT_LIMITS.get(tier, 10000)
