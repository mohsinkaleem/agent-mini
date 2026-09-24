"""Lightweight token estimation and model tier classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace


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

# Parameter count in the name: "8b", "0.6b", "8x7b" (MoE). The lookbehind
# skips version digits ("qwen2.5") and active-param tags ("30b-a3b").
_SIZE = re.compile(r"(?<![\w.])(?:(\d+)x)?(\d+(?:\.\d+)?)b(?![a-z0-9])")
_SIZE_MILLIONS = re.compile(r"(?<![\w.])\d+m(?![a-z0-9])")
_API = re.compile(r"^(?:gpt-|o\d|chatgpt|claude|gemini|grok|deepseek-(?:chat|reasoner|v\d))")
# Well-known names that carry no size tag.
_KNOWN_SIZELESS = (("phi-4-mini", "tiny"), ("phi4-mini", "tiny"), ("nemo", "medium"))

TIERS = ("tiny", "small", "medium", "large", "cloud")


def classify_model_tier(model_name: str) -> str:
    """Classify a model name into tiny / small / medium / large / cloud."""
    name = model_name.lower().rsplit("/", 1)[-1]
    m = _SIZE.search(name)
    if not m:
        if _API.search(name):
            return "cloud"
        if _SIZE_MILLIONS.search(name):
            return "tiny"
        for key, tier in _KNOWN_SIZELESS:
            if key in name:
                return tier
        return "small"  # safe default
    size = float(m.group(2)) * (int(m.group(1)) if m.group(1) else 1)
    return tier_for_size(size)


def tier_for_size(params_b: float) -> str:
    """Tier for a parameter count in billions."""
    if params_b < 4:
        return "tiny"
    if params_b < 9:
        return "small"
    if params_b < 20:
        return "medium"
    if params_b <= 72:
        return "large"
    return "cloud"


@dataclass(frozen=True)
class TierProfile:
    """Budgets derived from the model tier."""

    tier: str
    context: int  # effective context: where accuracy stays high (~90%)
    max_iterations: int
    output_limit: int  # max chars for a single tool output
    memory_items: int  # recent memories attached to the system prompt


_PROFILES = {
    "tiny": TierProfile("tiny", 3000, 10, 2000, 0),
    "small": TierProfile("small", 6000, 15, 4000, 3),
    "medium": TierProfile("medium", 12000, 20, 8000, 5),
    "large": TierProfile("large", 20000, 25, 20000, 5),
    "cloud": TierProfile("cloud", 32000, 25, 50000, 5),
}


def get_profile(
    model_name: str,
    agent_cfg: dict | None = None,
    params_b: float | None = None,
) -> TierProfile:
    """Return the tier profile for *model_name*, honouring ``agent.tier`` / ``agent.contextWindow``.

    *params_b* (the size the server reports) beats guessing from the name.
    """
    agent_cfg = agent_cfg or {}
    tier = agent_cfg.get("tier") or (
        tier_for_size(params_b) if params_b else classify_model_tier(model_name)
    )
    if tier not in _PROFILES:
        raise ValueError(f"Invalid agent.tier {tier!r}. Use one of: {', '.join(TIERS)}.")
    profile = _PROFILES[tier]
    if agent_cfg.get("contextWindow"):
        profile = replace(profile, context=int(agent_cfg["contextWindow"]))
    return profile


# Room for the system prompt and the model's reply on top of the conversation budget.
_PROMPT_RESERVE = 1024
_REPLY_RESERVE = 2048


def context_overhead(tool_defs: list[dict] | None = None) -> int:
    """Tokens needed on top of the conversation budget: tool schemas plus reserves."""
    schemas = estimate_tokens(json.dumps(tool_defs)) if tool_defs else 0
    return schemas + _PROMPT_RESERVE + _REPLY_RESERVE


def num_ctx_for(profile: TierProfile, tool_defs: list[dict] | None = None) -> int:
    """Context window to request from the server (Ollama ``num_ctx``), in tokens.

    Conversation budget + tool schemas + prompt and reply reserves, rounded up
    to a multiple of 2048.
    """
    need = profile.context + context_overhead(tool_defs)
    return -(-need // 2048) * 2048
