"""System prompt builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .memory import Memory

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


def build_system_prompt(config: dict, memory: Memory) -> str:
    """Render the full system prompt with live context."""
    workspace = Path(config.get("workspace", "~/.agent-mini/workspace")).expanduser()

    # Use safe substitution to avoid format string injection from user config
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        workspace=str(workspace),
    )

    custom = config.get("agent", {}).get("systemPrompt", "")
    if custom:
        # Append directly — no .format() call on user-controlled content
        prompt += f"\n## User instructions\n{custom}\n"

    recent = memory.get_recent(5)
    if recent:
        items = "\n".join(f"- {m['key']}: {m['value']}" for m in recent)
        prompt += f"\n## Recent memories\n{items}\n"

    return prompt.strip()
