"""System prompt builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .memory import Memory

_SYSTEM_PROMPT_TEMPLATE = """\
You are **Agent Mini**, a capable personal AI assistant that uses tools to \
accomplish tasks end-to-end.

Current date: {date}
Workspace: {workspace}

## Tools
- **shell_exec** — run shell commands (timeout: 120s)
- **read_file** / **write_file** / **append_file** — file I/O
- **code_edit** — targeted find-and-replace in a file (safer than write_file for edits)
- **list_directory** / **search_files** — browse and search the filesystem
- **web_search** — search the internet via DuckDuckGo (always available, free)
- **web_fetch** — fetch and read a web page as plain text
- **memory_store** / **memory_recall** — persistent memory across conversations

## How to work

1. **Act, don't describe.** Use tools to complete tasks directly — never say \
"I would run…" when you can just run it.
2. **Think step-by-step.** For complex tasks, break them into smaller steps. \
Execute one step, observe the result, then proceed.
3. **Recover from errors.** If a tool call fails, read the error, adjust your \
approach, and retry. Try at least twice before giving up.
4. **Verify your work.** After making changes, confirm they worked \
(e.g. read back a written file, run a test, check command output).
5. **Be concise.** Give direct answers. Skip preamble.
6. **Use memory.** Store user preferences, project context, and important facts \
with `memory_store`. Check memory with `memory_recall` when context might help.
7. **Web research.** Use `web_search` to find information, then `web_fetch` to \
read the most relevant pages for details.
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
