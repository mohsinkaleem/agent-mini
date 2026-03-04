"""System prompt builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .memory import Memory

_SYSTEM_PROMPT = """\
You are **Agent Mini**, a personal AI assistant.
You are helpful, concise, and capable of using tools to complete tasks.

Current date: {date}
Workspace: {workspace}

## Available tools
- **shell_exec** — run shell commands
- **read_file** / **write_file** / **append_file** — read and write files
- **list_directory** / **search_files** — browse and search the filesystem
- **web_search** — search the internet (if configured)
- **memory_store** / **memory_recall** — persistent memory across conversations

## Guidelines
- Be direct and helpful.
- Use tools proactively to complete tasks — don't just describe what you *would* do.
- Store important user preferences and facts in memory.
- When running shell commands, briefly explain what you're doing.
- If a task is ambiguous, ask a clarifying question.
{custom_prompt}
{memory_context}\
"""


def build_system_prompt(config: dict, memory: Memory) -> str:
    """Render the full system prompt with live context."""
    workspace = Path(config.get("workspace", "~/.agent-mini/workspace")).expanduser()

    custom = config.get("agent", {}).get("systemPrompt", "")
    if custom:
        custom = f"\n## User instructions\n{custom}\n"

    recent = memory.get_recent(5)
    memory_lines = ""
    if recent:
        items = "\n".join(f"- {m['key']}: {m['value']}" for m in recent)
        memory_lines = f"\n## Recent memories\n{items}\n"

    return _SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        workspace=str(workspace),
        custom_prompt=custom,
        memory_context=memory_lines,
    ).strip()
