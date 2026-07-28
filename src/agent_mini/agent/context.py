"""System prompt builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .memory import Memory
from .token_estimator import classify_model_tier

_SYSTEM_PROMPT_TEMPLATE = """\
You are Agent Mini, a personal AI assistant with tools for shell commands, \
file I/O, web search/fetch, and persistent memory.

Date: {date}
Workspace: {workspace}

<rules>
- ALWAYS read a file before editing it.
- Use tools to act directly — never say "I would run…" when you can just run it.
- If a tool fails, read the error and try a different approach. Retry at least twice.
- Verify changes (read back files, run tests, check output).
- Be concise. Skip preamble.
- Use memory_store/memory_recall for user preferences and project context.
</rules>
"""

# Compact rules for tiny models — every extra token trades against reasoning.
_TINY_RULES = """\
<rules>
- Read a file before editing it.
- Act with tools — don't describe, do.
- On tool error: read it, try a different approach.
- Verify changes.
- Be concise.
</rules>
"""

# How many recent memory entries to attach, per tier. Tiny models are
# easily confused by unrelated context; larger models benefit from more.
_MEMORY_BUDGET = {"tiny": 0, "small": 3, "medium": 5, "cloud": 5}


def _render_tool_list(tool_defs: list[dict] | None) -> str:
    """Render a compact one-line-per-tool list for small models to parse.

    Small models are more reliable when they can see an inline listing of
    available tool names + one-line descriptions in addition to the JSON
    schema. Skips silently if no defs supplied.
    """
    if not tool_defs:
        return ""
    lines = ["<available_tools>"]
    for td in tool_defs:
        func = td.get("function", {})
        name = func.get("name", "")
        desc = (func.get("description", "") or "").split("\n", 1)[0].strip()
        if name:
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    lines.append("</available_tools>")
    return "\n".join(lines)


def build_system_prompt(
    config: dict,
    memory: Memory,
    model_name: str | None = None,
    tool_defs: list[dict] | None = None,
) -> str:
    """Render the full system prompt with live context.

    When *model_name* is provided, the prompt is scaled to the model tier:
    tiny models get a compact rules block and no memory recall; larger
    tiers get the full rules and recent memory context.
    """
    workspace = Path(config.get("workspace", "~/.agent-mini/workspace")).expanduser()
    tier = classify_model_tier(model_name) if model_name else "small"

    # Tiny tier: minimal preamble + compact rules — every token trades
    # against the model's reasoning budget. Larger tiers get the full
    # descriptive template.
    if tier == "tiny":
        prompt = (
            "You are Agent Mini, a local AI assistant with tools.\n\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Workspace: {workspace}\n\n"
            + _TINY_RULES
        )
    else:
        # Use safe substitution to avoid format string injection from user config
        prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            workspace=str(workspace),
        )

    # Inline tool list — helps small models see available tools at a glance.
    tool_list = _render_tool_list(tool_defs)
    if tool_list:
        prompt += "\n" + tool_list + "\n"

    custom = config.get("agent", {}).get("systemPrompt", "")
    if custom:
        # Append directly — no .format() call on user-controlled content
        prompt += f"\n## User instructions\n{custom}\n"

    n_recent = _MEMORY_BUDGET.get(tier, 5)
    recent = memory.get_recent(n_recent) if n_recent else []
    if recent:
        items = "\n".join(f"- {m['key']}: {m['value']}" for m in recent)
        prompt += f"\n## Recent memories\n{items}\n"

    return prompt.strip()
