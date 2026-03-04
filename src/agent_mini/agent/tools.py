"""Built-in agent tools — shell, files, web search, memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from .memory import Memory

# ======================================================================
# Tool definitions  (OpenAI function-calling format)
# ======================================================================

TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a shell command and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative file path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Parent directories are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (defaults to workspace root).",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Brave Search API and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store important information in persistent memory for later recall. Use this to remember user preferences, facts, or anything worth keeping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short label/category for the memory.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The information to store.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Search persistent memory for previously stored information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for in memory.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ======================================================================
# Tool executor
# ======================================================================


class ToolExecutor:
    """Execute agent tools and return results as plain strings."""

    def __init__(self, config: dict, memory: Memory):
        self._config = config
        self._memory = memory
        self._workspace = Path(
            config.get("workspace", "~/.agent-mini/workspace")
        ).expanduser()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._restrict = config.get("tools", {}).get("restrictToWorkspace", False)
        self._search_key: str = (
            config.get("tools", {}).get("webSearch", {}).get("apiKey", "")
        )

    def get_tool_defs(self) -> list[dict]:
        """Return definitions for all *available* tools."""
        defs = [t for t in TOOL_DEFS if t["function"]["name"] != "web_search"]
        if self._search_key:
            defs.append(
                next(t for t in TOOL_DEFS if t["function"]["name"] == "web_search")
            )
        return defs

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(self, name: str, arguments: dict) -> str:
        """Run tool *name* with *arguments* and return a result string."""
        try:
            match name:
                case "shell_exec":
                    return await self._shell_exec(arguments["command"])
                case "read_file":
                    return self._read_file(arguments["path"])
                case "write_file":
                    return self._write_file(arguments["path"], arguments["content"])
                case "list_directory":
                    return self._list_directory(arguments.get("path", "."))
                case "web_search":
                    return await self._web_search(arguments["query"])
                case "memory_store":
                    return self._memory.store(arguments["key"], arguments["value"])
                case "memory_recall":
                    return self._memory.recall(arguments["query"])
                case _:
                    return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self._workspace / p
        if self._restrict:
            resolved = str(p.resolve())
            ws_resolved = str(self._workspace.resolve())
            if not resolved.startswith(ws_resolved):
                raise PermissionError(f"Access denied: {p} is outside workspace")
        return p

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _shell_exec(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: command timed out after 120 seconds"

        out = stdout.decode(errors="replace")
        if stderr:
            out += "\n" + stderr.decode(errors="replace")
        if len(out) > 50_000:
            out = out[:50_000] + "\n… (truncated)"
        return out or "(no output)"

    def _read_file(self, path: str) -> str:
        p = self._resolve_path(path)
        content = p.read_text(errors="replace")
        if len(content) > 100_000:
            content = content[:100_000] + "\n… (truncated)"
        return content

    def _write_file(self, path: str, content: str) -> str:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes → {p}"

    def _list_directory(self, path: str) -> str:
        p = self._resolve_path(path)
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines: list[str] = []
        for e in entries[:200]:
            prefix = "📁 " if e.is_dir() else "📄 "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) or "(empty directory)"

    async def _web_search(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 5},
                headers={
                    "X-Subscription-Token": self._search_key,
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[str] = []
        for r in data.get("web", {}).get("results", [])[:5]:
            results.append(
                f"**{r['title']}**\n{r['url']}\n{r.get('description', '')}"
            )
        return "\n\n".join(results) or "No results found."
