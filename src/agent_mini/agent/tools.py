"""""Built-in agent tools — shell, files, web search/fetch, memory."""

from __future__ import annotations

import asyncio
import json
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
    _tool("shell_exec", "Execute a shell command and return stdout+stderr.",
          {"command": _param("The shell command to run.")}, ["command"]),
    _tool("read_file", "Read the full contents of a file.",
          {"path": _param("Absolute or workspace-relative file path.")}, ["path"]),
    _tool("append_file", "Append content to a file (creates it if missing).",
          {"path": _param("File path."), "content": _param("Content to append.")},
          ["path", "content"]),
    _tool("write_file", "Create or overwrite a file. Parent dirs created automatically.",
          {"path": _param("File path."), "content": _param("File content.")},
          ["path", "content"]),
    _tool("list_directory", "List files and folders in a directory.",
          {"path": _param("Directory path (defaults to workspace root).", ".")}, []),
    _tool("search_files", "Search for text/regex in files. Returns matching lines with paths and line numbers.",
          {"query": _param("Search query (regex)."),
           "path": _param("Directory to search in.", ".")}, ["query"]),
    _tool("memory_store", "Store important information in persistent memory for later recall.",
          {"key": _param("Short label/category."),
           "value": _param("The information to store.")}, ["key", "value"]),
    _tool("memory_recall", "Search persistent memory for previously stored information.",
          {"query": _param("Keywords to search for.")}, ["query"]),
]

_WEB_TOOLS: list[dict] = [
    _tool("web_search", "Search the web using DuckDuckGo (free, no API key). Returns top results with titles, URLs, and snippets.",
          {"query": _param("Search query.")}, ["query"]),
    _tool("web_fetch", "Fetch a web page and return its content as readable plain text. Use after web_search to read full articles.",
          {"url": _param("The URL to fetch.")}, ["url"]),
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
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                    "li", "tr", "blockquote", "section", "article"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                    "li", "tr", "blockquote", "section", "article"):
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
        html, re.DOTALL,
    )
    if not blocks:
        # Fallback: try grabbing <a class="result__a"> directly
        blocks = re.findall(
            r'<div[^>]*class="[^"]*links_main[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*links_main|$)',
            html, re.DOTALL,
        )

    for block in blocks:
        # Extract URL from href in result__a or result-link
        url_match = re.search(r'href="([^"]+)"', block)
        title_match = re.search(r'class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
        snippet_match = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</[at]>', block, re.DOTALL)

        if not url_match:
            continue

        raw_url = url_match.group(1)
        # DuckDuckGo wraps URLs in a redirect — extract the real one
        uddg_match = re.search(r'[?&]uddg=([^&]+)', raw_url)
        url = unquote(uddg_match.group(1)) if uddg_match else raw_url

        # Skip DuckDuckGo internal links
        if url.startswith("/") or "duckduckgo.com" in url:
            continue

        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else url
        snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ""

        results.append({"title": title, "url": url, "snippet": snippet})

    return results[:8]


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
        self._http = httpx.AsyncClient(timeout=30)

    def get_tool_defs(self) -> list[dict]:
        """Return definitions for all *available* tools."""
        return list(_CORE_TOOLS) + list(_WEB_TOOLS)

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._http.aclose()

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

    def _list_directory(self, path: str) -> str:
        p = self._resolve_path(path)
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines: list[str] = []
        for e in entries[:200]:
            prefix = "📁 " if e.is_dir() else "📄 "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) or "(empty directory)"

    async def _search_files(self, query: str, path: str) -> str:
        root = self._resolve_path(path)

        # Prefer ripgrep, fall back to grep
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "--line-number", "--no-heading", "--hidden",
                   "--glob", "!.git", query, str(root)]
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
        resp = await self._http.get(url, headers=headers, follow_redirects=True, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
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
