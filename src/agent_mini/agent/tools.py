"""Built-in agent tools — shell, files, web search/fetch, memory."""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import importlib.util
import ipaddress
import logging
import os
import re
import shutil
import signal
from collections.abc import Awaitable, Callable
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ..config import agent_home
from ..providers.base import INVALID_ARGS_KEY
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


def _int_param(desc: str) -> dict:
    return {"type": "integer", "description": desc}


_TOOLS: list[dict] = [
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
        "Read a file. Returns text content. Use offset/limit for a line range of a big file.",
        {
            "path": _param("File path."),
            "offset": _int_param("First line to read (1-based). Optional."),
            "limit": _int_param("Max number of lines to read. Optional."),
        },
        ["path"],
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
        "find_files",
        "Find files by name/glob, recursively (e.g. '*.py', 'src/**/test_*.py').",
        {
            "pattern": _param("Glob pattern."),
            "path": _param("Directory to search.", "."),
        },
        ["pattern"],
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
    _tool(
        "memory_forget",
        "Delete a memory by its key.",
        {"key": _param("Key of the memory to delete.")},
        ["key"],
    ),
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

_BUILTIN_TOOL_NAMES = frozenset(d["function"]["name"] for d in _TOOLS)

# Directories that are never worth listing or searching.
_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
})

# Env vars that look like secrets are not passed to shell_exec.
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)

# Asked before running a tool listed in tools.confirm: (name, arguments) -> allowed?
Approver = Callable[[str, dict], Awaitable[bool]]


def _untrusted(source: str, text: str) -> str:
    """Mark external content so the model can tell data from instructions."""
    return f'<untrusted_content source="{source.replace(chr(34), "%22")}">\n{text}\n</untrusted_content>'


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _fuzzy_replace(content: str, old_text: str, new_text: str) -> tuple[str, int, int] | int:
    """Replace *old_text* matched line by line, ignoring leading/trailing whitespace.

    Returns ``(new_content, first_line, new_line_count)`` on a unique match,
    otherwise the number of matches (0 or >1). New lines are re-indented by
    the difference between the file's indentation and *old_text*'s.
    """
    lines = content.split("\n")
    old_lines = old_text.strip("\n").split("\n")
    key = [ln.strip() for ln in old_lines]
    if not any(key):
        return 0
    stripped = [ln.strip() for ln in lines]
    n = len(key)
    hits = [i for i in range(len(lines) - n + 1) if stripped[i : i + n] == key]
    if len(hits) != 1:
        return len(hits)
    i = hits[0]
    file_indent = next((_indent_of(ln) for ln in lines[i : i + n] if ln.strip()), "")
    old_indent = next((_indent_of(ln) for ln in old_lines if ln.strip()), "")
    new_lines = new_text.strip("\n").split("\n") if new_text.strip("\n") else []
    if file_indent != old_indent:
        new_lines = [
            file_indent + ln[len(old_indent):] if ln.strip() and ln.startswith(old_indent) else ln
            for ln in new_lines
        ]
    return "\n".join(lines[:i] + new_lines + lines[i + n :]), i, len(new_lines)


def _closest_match(content: str, old_text: str) -> str:
    """Describe the region of *content* most similar to *old_text*, if any is close."""
    lines = content.split("\n")
    old_lines = old_text.strip("\n").split("\n")
    n = len(old_lines)
    if len(lines) > 5000 or n > 200:
        return ""
    target = "\n".join(ln.strip() for ln in old_lines)
    best_ratio, best_i = 0.0, -1
    for i in range(max(1, len(lines) - n + 1)):
        matcher = difflib.SequenceMatcher(None, "\n".join(ln.strip() for ln in lines[i : i + n]), target)
        if matcher.real_quick_ratio() > best_ratio and matcher.quick_ratio() > best_ratio:
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
    if best_ratio < 0.6:
        return ""
    snippet = "\n".join(lines[best_i : best_i + n])
    return f"\nClosest match (line {best_i + 1}, {best_ratio:.0%} similar):\n{snippet}"


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill *proc* and every child it spawned (it must run in its own session)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


# ======================================================================
# HTML → plain text extractor (zero dependencies)
# ======================================================================


class _HTMLToText(HTMLParser):
    """Minimal HTML → readable text converter using stdlib only."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "section", "article",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "br" or tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
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


async def _check_url_ssrf(url: str) -> str | None:
    """Return an error string if *url* targets a private/loopback host.

    Cheap SSRF guard for web_fetch — blocks the common attack surface
    (localhost, RFC-1918, link-local, cloud metadata endpoints) without
    trying to be a full-blown proxy filter. Only ``http(s)`` allowed.
    Returns ``None`` when the URL passes the checks.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"scheme '{parsed.scheme or '?'}' is not allowed (use http/https)"

    host = parsed.hostname
    if not host:
        return "URL missing hostname"

    # Fast path: literal IP address
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"host {host} is a private/loopback/link-local address"
        return None
    except ValueError:
        pass

    # DNS resolution — check every A/AAAA record so a hostname pointing
    # into RFC-1918 is caught.
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as e:
        return f"cannot resolve host {host}: {e}"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"host {host} resolves to private/loopback address {addr}"
    return None


def _strip_tags(fragment: str) -> str:
    """Strip HTML tags and decode entities from a small fragment."""
    return unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _unwrap_ddg_url(raw: str) -> str:
    """DuckDuckGo wraps result links in a redirect — pull out the real URL."""
    match = re.search(r"[?&]uddg=([^&]+)", raw)
    return unquote(match.group(1)) if match else unescape(raw)


def _is_external(url: str) -> bool:
    """True when *url* is a real result rather than a DuckDuckGo internal link."""
    return url.startswith("http") and "duckduckgo.com" not in url


def _parse_ddg_html(page: str) -> list[dict[str, str]]:
    """Parse results from the html.duckduckgo.com endpoint (div markup)."""
    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[_ ]results_links[^"]*"[^>]*>(.*?)</div>\s*</div>',
        page,
        re.DOTALL,
    )
    if not blocks:
        # Fallback: split on the per-result container class
        blocks = re.findall(
            r'<div[^>]*class="[^"]*links_main[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*links_main|$)',
            page,
            re.DOTALL,
        )

    results: list[dict[str, str]] = []
    for block in blocks:
        url_match = re.search(r'href="([^"]+)"', block)
        if not url_match:
            continue
        url = _unwrap_ddg_url(url_match.group(1))
        if not _is_external(url):
            continue

        title_match = re.search(
            r'class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL
        )
        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</[at]>', block, re.DOTALL
        )
        results.append({
            "title": _strip_tags(title_match.group(1)) if title_match else url,
            "url": url,
            "snippet": _strip_tags(snippet_match.group(1)) if snippet_match else "",
        })
    return results[:8]


def _parse_ddg_lite(page: str) -> list[dict[str, str]]:
    """Parse results from the lite.duckduckgo.com endpoint (table markup)."""
    anchors = re.findall(
        r"<a\b([^>]*class=['\"]?result-link['\"]?[^>]*)>(.*?)</a>",
        page,
        re.DOTALL | re.IGNORECASE,
    )
    snippets = re.findall(
        r"class=['\"]?result-snippet['\"]?[^>]*>(.*?)</td>",
        page,
        re.DOTALL | re.IGNORECASE,
    )

    results: list[dict[str, str]] = []
    for index, (attrs, title_html) in enumerate(anchors):
        href = re.search(r"href=['\"]([^'\"]+)", attrs)
        if not href:
            continue
        url = _unwrap_ddg_url(href.group(1))
        if not _is_external(url):
            continue
        results.append({
            "title": _strip_tags(title_html) or url,
            "url": url,
            "snippet": _strip_tags(snippets[index]) if index < len(snippets) else "",
        })
    return results[:8]


# Markers DuckDuckGo serves instead of results when it rate-limits or
# challenges the client (common behind corporate proxies and VPNs).
_DDG_BLOCK_MARKERS = (
    "anomaly-modal",
    "unfortunately, bots use duckduckgo",
    "detected unusual activity",
    "challenge-form",
    "please try again later",
)

# Markers that mean "the page rendered fine, the query just had no hits".
_DDG_EMPTY_MARKERS = ("no results", "not many great matches")


def _classify_ddg_page(page: str) -> str:
    """Return ``blocked``, ``empty`` or ``ok`` for a DuckDuckGo response body."""
    head = page[:8000].lower()
    if any(marker in head for marker in _DDG_BLOCK_MARKERS):
        return "blocked"
    if any(marker in head for marker in _DDG_EMPTY_MARKERS):
        return "empty"
    return "ok"


# ======================================================================
# Tool executor
# ======================================================================


class ToolExecutor:
    """Execute agent tools and return results as plain strings."""

    _DEFAULT_BLOCKED_COMMANDS: dict[str, str] = {
        # rm with both a recursive and a force flag, in any order or spelling.
        r"\brm\b(?=[^|;&\n]*\s(?:-[a-z]*r|--recursive\b))(?=[^|;&\n]*\s(?:-[a-z]*f|--force\b))": (
            "destructive rm -rf"
        ),
        r"\bsudo\b": "sudo elevation",
        r"\bmkfs\b": "filesystem format",
        r"\bdd\s+if=": "raw disk write",
        r":\(\)\s*\{": "fork bomb",
        r"\b(chmod|chown)\s+(-R\s+)?[0-7]*\s+/[^\s]*$": "root permission change",
    }

    SANDBOX_LEVELS = ("unrestricted", "workspace", "readonly")

    # Tools blocked at each sandbox level
    _READONLY_BLOCKED_TOOLS = frozenset({"shell_exec", "write_file", "code_edit"})
    _MEMORY_TOOLS = frozenset({"memory_store", "memory_recall", "memory_forget"})
    _UNDO_LIMIT = 50

    def __init__(self, config: dict, memory: Memory):
        self._config = config
        self._memory = memory
        self._workspace = Path(
            config.get("workspace", "~/.agent-mini/workspace")
        ).expanduser()
        self._workspace.mkdir(parents=True, exist_ok=True)
        tools_cfg = config.get("tools", {})
        raw_level = tools_cfg.get("sandboxLevel", "workspace")
        level = str(raw_level).strip().lower()
        if level not in self.SANDBOX_LEVELS:
            raise ValueError(
                f"Invalid tools.sandboxLevel {raw_level!r}. "
                f"Use one of: {', '.join(self.SANDBOX_LEVELS)}."
            )
        self._sandbox_level = level
        # Only "unrestricted" may touch paths outside the workspace.
        self._restrict = level != "unrestricted" or bool(
            tools_cfg.get("restrictToWorkspace", False)
        )
        self._shell_timeout = float(tools_cfg.get("shellTimeout", 120))
        env_allow = set(tools_cfg.get("shellEnvAllow", []))
        self._shell_env = {
            k: v for k, v in os.environ.items()
            if k in env_allow or not _SECRET_ENV.search(k)
        }
        # Approval mode: tools listed here need a yes from self.approver.
        self._confirm = {str(t) for t in tools_cfg.get("confirm", [])}
        self.approver: Approver | None = None
        # (path, original bytes or None if the file didn't exist) per file change.
        self._undo: list[tuple[Path, bytes | None]] = []
        self._hidden_tools: set[str] = set()
        if level == "readonly":
            self._hidden_tools |= self._READONLY_BLOCKED_TOOLS
        if not config.get("memory", {}).get("enabled", True):
            self._hidden_tools |= self._MEMORY_TOOLS
        self._http = httpx.AsyncClient(timeout=30)
        # Command blocklist — user config extends (not replaces) defaults
        user_blocked = tools_cfg.get("blockedCommands", [])
        all_patterns = list(self._DEFAULT_BLOCKED_COMMANDS.keys()) + user_blocked
        self._blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in all_patterns
        ]
        # Load plugins
        self._plugins: dict[str, dict] = {}  # name → {definition, handler}
        self._load_plugins()

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def sandbox_level(self) -> str:
        return self._sandbox_level

    @property
    def confirm_tools(self) -> set[str]:
        return set(self._confirm)

    def _load_plugins(self) -> None:
        """Discover and load plugins from ~/.agent-mini/plugins/."""
        plugins_dir = agent_home() / "plugins"
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
                    if name in _BUILTIN_TOOL_NAMES:
                        log.warning("Skipping plugin %s: name '%s' collides with a built-in tool", py_file, name)
                        continue
                    readonly = bool(tool_def.get("x-readonly", False))
                    if self._sandbox_level == "readonly" and not readonly:
                        log.info("Plugin %s disabled in readonly mode (no x-readonly flag)", name)
                        continue
                    self._plugins[name] = {
                        # Providers reject unknown keys in tool definitions.
                        "definition": {k: v for k, v in tool_def.items() if k != "x-readonly"},
                        "handler": handler,
                    }
                    log.info("Loaded plugin: %s from %s", name, py_file)
            except Exception as e:
                log.warning("Failed to load plugin %s: %s", py_file, e)

    def get_tool_defs(self) -> list[dict]:
        """Return definitions for all *available* tools (built-in + plugins)."""
        defs = [d for d in _TOOLS if d["function"]["name"] not in self._hidden_tools]
        for plugin in self._plugins.values():
            defs.append(plugin["definition"])
        return defs

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _check_arguments(self, name: str, arguments: dict) -> str | None:
        """Return an ``Error:`` string the model can act on, or None if the call is well-formed."""
        defs = {d["function"]["name"]: d for d in self.get_tool_defs()}
        if name not in defs:
            return f"Error: unknown tool '{name}'. Available: {', '.join(defs)}"
        if INVALID_ARGS_KEY in arguments:
            return (
                f"Error: arguments for {name} were not valid JSON: "
                f"{str(arguments[INVALID_ARGS_KEY])[:200]}. Re-emit the call with valid JSON."
            )
        params = defs[name]["function"].get("parameters", {})
        required = params.get("required", [])
        missing = [r for r in required if r not in arguments]
        if missing:
            props = params.get("properties", {})
            expected = ", ".join(
                f'"{r}": <{props.get(r, {}).get("type", "string")}>' for r in required
            )
            names = ", ".join(f"'{m}'" for m in missing)
            return (
                f"Error: missing required argument {names} for {name}. "
                f"Expected {{{expected}}}"
            )
        return None

    async def execute(self, name: str, arguments: dict) -> str:
        """Run tool *name* with *arguments* and return a result string."""
        # Sandbox level enforcement
        if self._sandbox_level == "readonly" and name in self._READONLY_BLOCKED_TOOLS:
            return (
                f"Error: tool '{name}' is blocked in readonly sandbox mode. "
                "Only read, search, memory, and web tools are allowed."
            )
        invalid = self._check_arguments(name, arguments)
        if invalid:
            return invalid
        if name in self._confirm:
            if self.approver is None:
                return (
                    f"Error: {name} needs user approval (tools.confirm), which is not "
                    "available here. Tell the user what you wanted to do instead."
                )
            if not await self.approver(name, arguments):
                return (
                    f"Error: the user declined this {name} call. Do not retry it; "
                    "ask the user how to proceed or take a different approach."
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
                    return self._read_file(
                        arguments["path"], arguments.get("offset"), arguments.get("limit")
                    )
                case "write_file":
                    return self._write_file(arguments["path"], arguments["content"])
                case "list_directory":
                    return self._list_directory(arguments.get("path", "."))
                case "search_files":
                    return await self._search_files(
                        arguments["query"], arguments.get("path", ".")
                    )
                case "find_files":
                    return self._find_files(arguments["pattern"], arguments.get("path", "."))
                case "web_search":
                    return await self._web_search(arguments["query"])
                case "web_fetch":
                    return await self._web_fetch(arguments["url"])
                case "memory_store":
                    return self._memory.store(arguments["key"], arguments["value"])
                case "memory_recall":
                    return self._memory.recall(arguments["query"])
                case "memory_forget":
                    return self._memory.forget(arguments["key"])
                case _:
                    handler = self._plugins[name]["handler"]
                    result = handler(arguments)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result)
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

    def _display(self, p: Path) -> str:
        """Workspace-relative path when possible (saves tokens)."""
        ws = self._workspace.resolve()
        return str(p.relative_to(ws)) if p.is_relative_to(ws) else str(p)

    # ------------------------------------------------------------------
    # Undo and previews (used by the CLI)
    # ------------------------------------------------------------------

    def _checkpoint(self, p: Path) -> None:
        """Remember *p*'s current content so undo() can restore it."""
        self._undo.append((p, p.read_bytes() if p.exists() else None))
        del self._undo[: -self._UNDO_LIMIT]

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    def undo(self) -> str:
        """Revert the most recent file change made by write_file or code_edit."""
        if not self._undo:
            return "Nothing to undo."
        p, original = self._undo.pop()
        if original is None:
            p.unlink(missing_ok=True)
            return f"Deleted {self._display(p)} (the agent created it)."
        p.write_bytes(original)
        return f"Restored {self._display(p)}."

    def preview_change(self, name: str, arguments: dict) -> str | None:
        """Unified diff of what a file-writing tool call would change, or None for other tools."""
        if name not in ("write_file", "code_edit"):
            return None
        try:
            p = self._resolve_path(arguments.get("path", ""))
            old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if name == "write_file":
                new = arguments.get("content", "")
            else:
                planned = self._plan_edit(p, arguments.get("old_text", ""), arguments.get("new_text", ""))
                if isinstance(planned, str):
                    return planned
                new = planned[0]
        except Exception as e:
            return f"(no preview: {e})"
        shown = self._display(p)
        diff = difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{shown}", f"b/{shown}"
        )
        return "".join(diff) or "(no changes)"

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
            env=self._shell_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._shell_timeout
            )
        except TimeoutError:
            await _kill_process_group(proc)
            return f"Error: command timed out after {self._shell_timeout:g} seconds"

        out = stdout.decode(errors="replace")
        if stderr:
            out += ("\n" if out else "") + stderr.decode(errors="replace")
        out = out or "(no output)"
        # No "Error:" prefix: grep exiting 1 on no match is not a failure.
        if proc.returncode:
            out += f"\n[exit code: {proc.returncode}]"
        return out

    def _read_file(self, path: str, offset=None, limit=None) -> str:
        p = self._resolve_path(path)
        with open(p, "rb") as f:
            head = f.read(8192)
        if b"\0" in head:
            return f"Error: {self._display(p)} is a binary file ({p.stat().st_size} bytes)."
        content = p.read_text(encoding="utf-8", errors="replace")
        offset = int(offset) if offset not in (None, "") else None
        limit = int(limit) if limit not in (None, "") else None
        if offset is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start = max(1, offset or 1)
            if lines and start > len(lines):
                return f"Error: offset {start} is past the end of the file ({len(lines)} lines)."
            end = len(lines) if limit is None else min(len(lines), start - 1 + max(1, limit))
            content = f"[lines {start}-{end} of {len(lines)}]\n" + "".join(lines[start - 1 : end])
        return content

    def _write_file(self, path: str, content: str) -> str:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint(p)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars → {self._display(p)}"

    def _plan_edit(
        self, p: Path, old_text: str, new_text: str
    ) -> str | tuple[str, int, int, bool, bool]:
        """Work out a code_edit without writing it.

        Returns an ``Error:`` string, or ``(new_content, first_line,
        new_line_count, fuzzy, crlf)`` where *new_content* uses ``\\n``.
        """
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: {self._display(p)} is not valid UTF-8; refusing to edit it."
        crlf = "\r\n" in raw
        content = raw.replace("\r\n", "\n")
        old = old_text.replace("\r\n", "\n")
        new = new_text.replace("\r\n", "\n")
        count = content.count(old)
        if count > 1:
            return (
                f"Error: old_text matches {count} locations. "
                "Include more surrounding lines so it matches exactly once."
            )
        if count == 1:
            start = content.index(old)
            result = content[:start] + new + content[start + len(old):]
            return result, content.count("\n", 0, start), new.count("\n") + 1, False, crlf
        match = _fuzzy_replace(content, old, new)
        if isinstance(match, int):
            if match > 1:
                return (
                    f"Error: old_text matches {match} locations (ignoring whitespace). "
                    "Include more surrounding lines so it matches exactly once."
                )
            return (
                "Error: old_text not found in file. Copy it exactly from read_file output, "
                "including indentation." + _closest_match(content, old)
            )
        result, first, n_new = match
        return result, first, n_new, True, crlf

    def _code_edit(self, path: str, old_text: str, new_text: str) -> str:
        p = self._resolve_path(path)
        shown = self._display(p)
        if not p.exists():
            return f"Error: file not found: {shown}"
        if not old_text:
            return "Error: old_text is empty. Use write_file to create or rewrite the file."
        planned = self._plan_edit(p, old_text, new_text)
        if isinstance(planned, str):
            return planned
        content, first, n_new, fuzzy, crlf = planned
        self._checkpoint(p)
        p.write_bytes((content.replace("\n", "\r\n") if crlf else content).encode("utf-8"))

        # Showing the result saves the model a read_file call to verify it.
        lines = content.split("\n")
        lo, hi = max(0, first - 2), min(len(lines), first + max(n_new, 1) + 2)
        note = " (matched ignoring whitespace)" if fuzzy else ""
        summary = f"Edited {shown}: replaced 1 occurrence{note}."
        if hi - lo > 30:
            return summary
        return f"{summary} Lines {lo + 1}-{hi} now:\n" + "\n".join(lines[lo:hi])

    def _list_directory(self, path: str) -> str:
        p = self._resolve_path(path)
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines: list[str] = []
        for e in entries[:200]:
            prefix = "[dir]  " if e.is_dir() else "[file] "
            lines.append(f"{prefix}{e.name}")
        if len(entries) > 200:
            lines.append(f"… and {len(entries) - 200} more entries")
        return "\n".join(lines) or "(empty directory)"

    def _find_files(self, pattern: str, path: str) -> str:
        root = self._resolve_path(path)
        pattern = pattern.strip()
        if pattern.startswith("./"):
            pattern = pattern[2:]
        # "*.py" matches at any depth; "**/x" also matches x at the top level.
        name_only = "/" not in pattern
        tail = pattern[3:] if pattern.startswith("**/") else None
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(root)
            for fn in sorted(filenames):
                rel = (rel_dir / fn).as_posix()
                if (
                    fnmatch.fnmatch(rel, pattern)
                    or (tail and fnmatch.fnmatch(rel, tail))
                    or (name_only and fnmatch.fnmatch(fn, pattern))
                ):
                    found.append(self._display(Path(dirpath) / fn))
            if len(found) > 10_000:
                break
        if not found:
            return f"No files matching '{pattern}'."
        found.sort()
        more = f"\n… and {len(found) - 200} more" if len(found) > 200 else ""
        return "\n".join(found[:200]) + more

    async def _search_files(self, query: str, path: str) -> str:
        root = self._resolve_path(path)
        # Relative paths in the output save tokens.
        ws = self._workspace.resolve()
        target = str(root.relative_to(ws)) if root.is_relative_to(ws) else str(root)

        # Prefer ripgrep, fall back to grep. "-e" and "--" stop a query
        # starting with "-" from being parsed as a flag.
        rg = shutil.which("rg")
        if rg:
            cmd = [
                rg,
                "--line-number",
                "--no-heading",
                "--hidden",
                "--max-columns", "300",
                "--glob", "!.git",
                "--glob", "!node_modules",
                "--glob", "!.venv",
                "-e", query,
                "--", target,
            ]
        else:
            grep = shutil.which("grep") or "grep"
            cmd = [
                grep, "-rnI",
                "--exclude-dir=.git",
                "--exclude-dir=node_modules",
                "--exclude-dir=.venv",
                "-e", query,
                "--", target,
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ws),
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            await _kill_process_group(proc)
            return "Error: search timed out after 30 seconds. Narrow the path or pattern."

        if proc.returncode not in (0, 1):
            err = stderr.decode(errors="replace").strip()
            return f"Error: search failed: {err or 'unknown error'}"

        out = stdout.decode(errors="replace").strip()
        return out or "No matches found."

    _BROWSER_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Tried in order — the lite endpoint is smaller and survives rate
    # limiting more often than the full HTML one.
    _DDG_ENDPOINTS = (
        ("https://lite.duckduckgo.com/lite/", _parse_ddg_lite),
        ("https://html.duckduckgo.com/html/", _parse_ddg_html),
    )

    async def _web_search(self, query: str) -> str:
        """Search via DuckDuckGo HTML — free, no API key required.

        Failures return an ``Error:`` string that names the cause and tells
        the model not to retry, otherwise small models loop on the same
        query until they burn through max_iterations.
        """
        failures: list[str] = []

        for url, parser in self._DDG_ENDPOINTS:
            host = urlparse(url).hostname or url
            try:
                resp = await self._http.post(
                    url,
                    data={"q": query},
                    headers={
                        "User-Agent": self._BROWSER_UA,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    follow_redirects=True,
                    timeout=15,
                )
            except httpx.HTTPError as e:
                failures.append(f"{host}: {type(e).__name__}")
                continue

            if resp.status_code >= 400:
                failures.append(f"{host}: HTTP {resp.status_code}")
                continue

            verdict = _classify_ddg_page(resp.text)
            if verdict == "blocked":
                failures.append(f"{host}: rate-limited or blocked by a network filter")
                continue

            results = parser(resp.text)
            if results:
                return _untrusted("web_search", "\n\n".join(
                    f"**{r['title']}**\n{r['url']}"
                    + (f"\n{r['snippet']}" if r["snippet"] else "")
                    for r in results
                ))

            if verdict == "empty":
                return (
                    f"No results found for '{query}'. "
                    "Try different or broader keywords."
                )
            failures.append(f"{host}: response had no parsable results")

        return (
            f"Error: web_search is unavailable ({'; '.join(failures)}). "
            "Do NOT retry this tool — it will keep failing. Either call "
            "web_fetch on a specific URL you already know, or answer from "
            "your own knowledge and tell the user the information may be "
            "out of date."
        )

    _MAX_REDIRECTS = 5
    _MAX_FETCH_BYTES = 2_000_000

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL and return readable text content.

        Blocks requests to private/loopback/link-local addresses to guard
        against SSRF (e.g. cloud metadata at 169.254.169.254, localhost
        services, RFC-1918 nets). Redirects are followed by hand so every
        hop is checked before it is requested. Best-effort — DNS rebinding
        can still bypass it, but this stops the common cases.
        """
        headers = {
            "User-Agent": self._BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        for hop in range(self._MAX_REDIRECTS + 1):
            blocked = await _check_url_ssrf(url)
            if blocked:
                return f"Error: {blocked}" + (" (after redirect)" if hop else "")
            async with self._http.stream("GET", url, headers=headers, timeout=20) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    url = str(resp.url.join(resp.headers["location"]))
                    continue
                if resp.status_code >= 400:
                    return f"Error: HTTP {resp.status_code} fetching {url}"
                content_type = resp.headers.get("content-type", "").lower()
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= self._MAX_FETCH_BYTES:
                        break
                body = b"".join(chunks)[: self._MAX_FETCH_BYTES].decode(
                    resp.encoding or "utf-8", errors="replace"
                )
            break
        else:
            return f"Error: too many redirects (more than {self._MAX_REDIRECTS})"

        # Non-HTML content (JSON, plain text, etc.) — return as-is
        if "html" not in content_type:
            return _untrusted(url, body)

        text = _html_to_text(body)
        if not text:
            return "(page returned no readable text)"
        return _untrusted(url, text)
