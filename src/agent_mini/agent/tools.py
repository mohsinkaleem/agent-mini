""" ""Built-in agent tools — shell, files, web search/fetch, memory."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import httpx

from .memory import Memory

log = logging.getLogger("agent-mini")

# ======================================================================
# Tool definitions  (OpenAI function-calling format)
# ======================================================================


def _tool(name: str, description: str, params: dict, required: list[str]) -> dict:
    """Shorthand for building an OpenAI function-calling tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            },
        },
    }


def _param(desc: str, default: str | None = None) -> dict:
    """Shorthand for a string parameter."""
    p: dict = {"type": "string", "description": desc}
    if default is not None:
        p["default"] = default
    return p


_CORE_TOOLS: list[dict] = [
    _tool(
        "shell_exec",
        "Run a shell command. Returns stdout+stderr.",
        {"command": _param("Shell command.")},
        ["command"],
    ),
    _tool(
        "code_edit",
        "Find-and-replace in a file. old_text must match exactly once.",
        {
            "path": _param("File path."),
            "old_text": _param("Exact text to find (must match once)."),
            "new_text": _param("Replacement text."),
        },
        ["path", "old_text", "new_text"],
    ),
    _tool(
        "read_file",
        "Read a file. Returns text content.",
        {"path": _param("File path.")},
        ["path"],
    ),
    _tool(
        "append_file",
        "Append text to a file. Creates it if missing.",
        {"path": _param("File path."), "content": _param("Text to append.")},
        ["path", "content"],
    ),
    _tool(
        "write_file",
        "Create or overwrite a file.",
        {"path": _param("File path."), "content": _param("File content.")},
        ["path", "content"],
    ),
    _tool(
        "list_directory",
        "List files and folders in a directory.",
        {"path": _param("Directory path.", ".")},
        [],
    ),
    _tool(
        "search_files",
        "Grep for text/regex in files. Returns matching lines.",
        {
            "query": _param("Regex pattern."),
            "path": _param("Directory to search.", "."),
        },
        ["query"],
    ),
    _tool(
        "memory_store",
        "Save a fact to persistent memory.",
        {
            "key": _param("Short label."),
            "value": _param("Information to store."),
        },
        ["key", "value"],
    ),
    _tool(
        "memory_recall",
        "Search persistent memory.",
        {"query": _param("Search keywords.")},
        ["query"],
    ),
]

_WEB_TOOLS: list[dict] = [
    _tool(
        "web_search",
        "Search the web via DuckDuckGo. Returns titles, URLs, snippets.",
        {"query": _param("Search query.")},
        ["query"],
    ),
    _tool(
        "web_fetch",
        "Fetch a URL as plain text.",
        {"url": _param("URL to fetch.")},
        ["url"],
    ),
]


# ======================================================================
# HTML → plain text extractor (zero dependencies)
# ======================================================================


class _HTMLToText(HTMLParser):
    """Minimal HTML → readable text converter using stdlib only."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in (
            "br",
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "blockquote",
            "section",
            "article",
        ):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in (
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "blockquote",
            "section",
            "article",
        ):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML."""
    parser = _HTMLToText()
    parser.feed(html)
    return parser.get_text()


def _parse_ddg_html(html: str) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results into structured results."""
    results: list[dict[str, str]] = []

    # DuckDuckGo HTML results have <a class="result__a"> for titles/URLs
    # and <a class="result__snippet"> for descriptions.
    # We use regex for reliability — no extra dependencies.
    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[_ ]results_links[^"]*"[^>]*>(.*?)</div>\s*</div>',
        html,
        re.DOTALL,
    )
    if not blocks:
        # Fallback: try grabbing <a class="result__a"> directly
        blocks = re.findall(
            r'<div[^>]*class="[^"]*links_main[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*links_main|$)',
            html,
            re.DOTALL,
        )

    for block in blocks:
        # Extract URL from href in result__a or result-link
        url_match = re.search(r'href="([^"]+)"', block)
        title_match = re.search(
            r'class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL
        )
        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</[at]>', block, re.DOTALL
        )

        if not url_match:
            continue

        raw_url = url_match.group(1)
        # DuckDuckGo wraps URLs in a redirect — extract the real one
        uddg_match = re.search(r"[?&]uddg=([^&]+)", raw_url)
        url = unquote(uddg_match.group(1)) if uddg_match else raw_url

        # Skip DuckDuckGo internal links
        if url.startswith("/") or "duckduckgo.com" in url:
            continue

        title = (
            re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else url
        )
        snippet = (
            re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
            if snippet_match
            else ""
        )

        results.append({"title": title, "url": url, "snippet": snippet})

    return results[:8]


# ======================================================================
# Tool executor
# ======================================================================


class ToolExecutor:
    """Execute agent tools and return results as plain strings."""

    _DEFAULT_BLOCKED_COMMANDS: dict[str, str] = {
        r"\brm\s+-[^\s]*r[^\s]*f": "destructive rm -rf",
        r"\bsudo\b": "sudo elevation",
        r"\bmkfs\b": "filesystem format",
        r"\bdd\s+if=": "raw disk write",
        r":\(\)\s*\{": "fork bomb",
        r"\b(chmod|chown)\s+(-R\s+)?[0-7]*\s+/[^\s]*$": "root permission change",
    }

    # Tools blocked at each sandbox level
    _READONLY_BLOCKED_TOOLS = frozenset({
        "shell_exec", "write_file", "append_file", "code_edit",
    })

    def __init__(self, config: dict, memory: Memory):
        self._config = config
        self._memory = memory
        self._workspace = Path(
            config.get("workspace", "~/.agent-mini/workspace")
        ).expanduser()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._restrict = config.get("tools", {}).get("restrictToWorkspace", False)
        self._sandbox_level = config.get("tools", {}).get("sandboxLevel", "workspace")
        # "workspace" level implies path restriction
        if self._sandbox_level == "workspace":
            self._restrict = True
        self._http = httpx.AsyncClient(timeout=30)
        # Command blocklist — user config extends (not replaces) defaults
        user_blocked = config.get("tools", {}).get("blockedCommands", [])
        all_patterns = list(self._DEFAULT_BLOCKED_COMMANDS.keys()) + user_blocked
        self._blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in all_patterns
        ]
        # Load plugins
        self._plugins: dict[str, dict] = {}  # name → {definition, handler}
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Discover and load plugins from ~/.agent-mini/plugins/."""
        plugins_dir = Path.home() / ".agent-mini" / "plugins"
        if not plugins_dir.is_dir():
            return
        for py_file in sorted(plugins_dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"agent_mini_plugin_{py_file.stem}", py_file
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Each plugin must export TOOL_DEF (dict) and handler (async callable)
                tool_def = getattr(module, "TOOL_DEF", None)
                handler = getattr(module, "handler", None)
                if tool_def and handler:
                    name = tool_def.get("function", {}).get("name", py_file.stem)
                    self._plugins[name] = {
                        "definition": tool_def,
                        "handler": handler,
                    }
                    log.info("Loaded plugin: %s from %s", name, py_file)
            except Exception as e:
                log.warning("Failed to load plugin %s: %s", py_file, e)

    def get_tool_defs(self) -> list[dict]:
        """Return definitions for all *available* tools (built-in + plugins)."""
        defs = list(_CORE_TOOLS) + list(_WEB_TOOLS)
        for plugin in self._plugins.values():
            defs.append(plugin["definition"])
        return defs

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(self, name: str, arguments: dict) -> str:
        """Run tool *name* with *arguments* and return a result string."""
        # Sandbox level enforcement
        if self._sandbox_level == "readonly" and name in self._READONLY_BLOCKED_TOOLS:
            return (
                f"Error: tool '{name}' is blocked in readonly sandbox mode. "
                "Only read, search, memory, and web tools are allowed."
            )
        try:
            match name:
                case "shell_exec":
                    return await self._shell_exec(arguments["command"])
                case "code_edit":
                    return self._code_edit(
                        arguments["path"],
                        arguments["old_text"],
                        arguments["new_text"],
                    )
                case "read_file":
                    return self._read_file(arguments["path"])
                case "append_file":
                    return self._append_file(arguments["path"], arguments["content"])
                case "write_file":
                    return self._write_file(arguments["path"], arguments["content"])
                case "list_directory":
                    return self._list_directory(arguments.get("path", "."))
                case "search_files":
                    return await self._search_files(
                        arguments["query"], arguments.get("path", ".")
                    )
                case "web_search":
                    return await self._web_search(arguments["query"])
                case "web_fetch":
                    return await self._web_fetch(arguments["url"])
                case "memory_store":
                    return self._memory.store(arguments["key"], arguments["value"])
                case "memory_recall":
                    return self._memory.recall(arguments["query"])
                case _:
                    # Check plugins before giving up
                    if name in self._plugins:
                        handler = self._plugins[name]["handler"]
                        result = handler(arguments)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return str(result)
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
        p = p.resolve()
        if self._restrict and not p.is_relative_to(self._workspace.resolve()):
            raise PermissionError(f"Access denied: {p} is outside workspace")
        return p

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _shell_exec(self, command: str) -> str:
        # Check command against blocklist
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                rule_name = self._DEFAULT_BLOCKED_COMMANDS.get(pattern.pattern, "custom rule")
                return (
                    f"Error: command blocked by security policy ({rule_name}). "
                    f"If this is intentional, adjust tools.blockedCommands in config."
                )
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

    def _append_file(self, path: str, content: str) -> str:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} bytes → {p}"

    def _code_edit(self, path: str, old_text: str, new_text: str) -> str:
        p = self._resolve_path(path)
        if not p.exists():
            return f"Error: file not found: {p}"
        content = p.read_text(errors="replace")
        count = content.count(old_text)
        if count == 0:
            return "Error: old_text not found in file. Check for exact match including whitespace."
        if count > 1:
            return f"Error: old_text matches {count} locations. Make it more specific to match exactly once."
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content)
        return f"Edited {p} — replaced 1 occurrence ({len(old_text)} → {len(new_text)} chars)"

    def _list_directory(self, path: str) -> str:
        p = self._resolve_path(path)
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines: list[str] = []
        for e in entries[:200]:
            prefix = "[dir]  " if e.is_dir() else "[file] "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) or "(empty directory)"

    async def _search_files(self, query: str, path: str) -> str:
        root = self._resolve_path(path)

        # Prefer ripgrep, fall back to grep
        rg = shutil.which("rg")
        if rg:
            cmd = [
                rg,
                "--line-number",
                "--no-heading",
                "--hidden",
                "--glob",
                "!.git",
                query,
                str(root),
            ]
        else:
            grep = shutil.which("grep") or "grep"
            cmd = [grep, "-rn", "--include=*", query, str(root)]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode not in (0, 1):
            err = stderr.decode(errors="replace").strip()
            return f"Error: search failed: {err or 'unknown error'}"

        out = stdout.decode(errors="replace").strip()
        if not out:
            return "No matches found."
        if len(out) > 100_000:
            out = out[:100_000] + "\n… (truncated)"
        return out

    _BROWSER_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    async def _web_search(self, query: str) -> str:
        """Search via DuckDuckGo HTML — free, no API key required."""
        resp = await self._http.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": self._BROWSER_UA,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            follow_redirects=True,
            timeout=15,
        )
        resp.raise_for_status()

        results = _parse_ddg_html(resp.text)
        if not results:
            return "No results found."

        lines: list[str] = []
        for r in results:
            entry = f"**{r['title']}**\n{r['url']}"
            if r.get("snippet"):
                entry += f"\n{r['snippet']}"
            lines.append(entry)
        return "\n\n".join(lines)

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL and return readable text content."""
        headers = {
            "User-Agent": self._BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = await self._http.get(
            url, headers=headers, follow_redirects=True, timeout=20
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()
        body = resp.text

        # Non-HTML content (JSON, plain text, etc.) — return as-is
        if "html" not in content_type:
            if len(body) > 100_000:
                body = body[:100_000] + "\n… (truncated)"
            return body

        text = _html_to_text(body)
        if len(text) > 80_000:
            text = text[:80_000] + "\n… (truncated)"
        if not text:
            return "(page returned no readable text)"
        return text
