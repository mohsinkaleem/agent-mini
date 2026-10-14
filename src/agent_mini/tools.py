"""Tools the model can call: shell, files, web, memory and plugins, behind one registry."""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import web
from .config import agent_home, with_defaults
from .memory import Memory
from .providers.base import INVALID_ARGS_KEY

log = logging.getLogger("agent-mini")

# Asked before running a tool listed in tools.confirm: (name, arguments) -> allowed?
Approver = Callable[[str, dict], Awaitable[bool]]

# Names models use for our tools (and the names before 0.5). They also keep
# old tools.confirm lists working.
TOOL_ALIASES = {
    "shell_exec": "shell", "bash": "shell", "run_shell": "shell", "run_command": "shell",
    "execute_command": "shell", "terminal": "shell",
    "code_edit": "edit_file", "str_replace": "edit_file", "replace_in_file": "edit_file",
    "create_file": "write_file", "list_directory": "list_files", "find_files": "list_files",
    "ls": "list_files", "grep": "search_files", "memory_store": "remember", "memory_recall": "recall",
}
# Argument names small models use instead of ours. Applied only when the tool
# has no parameter of that name.
_ARG_ALIASES = {
    "file_path": "path", "filepath": "path", "filename": "path", "file": "path", "directory": "path",
    "dir": "path", "cmd": "command", "old_string": "old_text", "new_string": "new_text",
    "old_str": "old_text", "new_str": "new_text", "contents": "content", "text": "content",
    "glob": "pattern", "pattern": "query", "regex": "query", "search": "query",
}
# Directories that are never worth listing recursively or searching.
_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
                        ".mypy_cache", ".ruff_cache", ".tox"})
# Env vars that look like secrets are not passed to the shell.
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_BLOCKED_COMMANDS = {
    # rm with both a recursive and a force flag, in any order or spelling.
    r"\brm\b(?=[^|;&\n]*\s(?:-[a-z]*r|--recursive\b))(?=[^|;&\n]*\s(?:-[a-z]*f|--force\b))": "destructive rm -rf",
    r"\bsudo\b": "sudo elevation",
    r"\bmkfs\b": "filesystem format",
    r"\bdd\s+if=": "raw disk write",
    r":\(\)\s*\{": "fork bomb",
    r"\b(chmod|chown)\s+(-R\s+)?[0-7]*\s+/[^\s]*$": "root permission change",
}
_UNDO_LIMIT = 50


@dataclass(frozen=True)
class Tool:
    definition: dict  # OpenAI function-calling schema
    call: Callable[[dict], str | Awaitable[str]]

    @property
    def name(self) -> str:
        return self.definition["function"]["name"]

    @property
    def params(self) -> dict:
        return self.definition["function"].get("parameters") or {}


def tool(description: str, writes: bool = False, **docs: str):
    """Mark a ``Tools`` method as a tool. Its signature gives the parameters; *docs* describe them.

    *writes*: it changes files or runs commands, so the readonly sandbox leaves it out.
    """
    def mark(fn):
        fn.__tool__ = (description, writes, docs)
        return fn
    return mark


def _schema(fn: Callable, description: str, docs: dict) -> dict:
    props, required = {}, []
    for p in list(inspect.signature(fn).parameters.values())[1:]:  # skip self
        kind = "integer" if "int" in str(p.annotation) else "string"
        props[p.name] = {"type": kind, "description": docs.get(p.name, "")}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


def _normalize(params: dict, arguments: dict) -> dict:
    """Map aliased argument names and coerce values to the declared types."""
    props = params.get("properties") or {}
    out: dict = {}
    for key, value in arguments.items():
        if value is None:  # "new_text": null means "not given", never the text "None"
            continue
        if key not in props and _ARG_ALIASES.get(key) in props:
            key = _ARG_ALIASES[key]
        if key in out:
            continue
        kind = props.get(key, {}).get("type")
        if kind == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        elif kind == "string" and not isinstance(value, str):
            value = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
        out[key] = value
    return out


# ── edit_file helpers ────────────────────────────────────────────────


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _base_indent(lines: list[str]) -> str:
    """The shortest indentation among the non-blank lines."""
    return min((_indent_of(ln) for ln in lines if ln.strip()), key=len, default="")


def _fuzzy_replace(content: str, old: str, new: str) -> tuple[str, int, int] | int:
    """Replace *old* matched line by line ignoring surrounding whitespace.

    Returns ``(new_content, first_line, new_line_count)`` on a unique match,
    otherwise the number of matches. *new* is re-indented to the file's indentation.
    """
    lines = content.split("\n")
    old_lines = old.strip("\n").split("\n")
    key = [ln.strip() for ln in old_lines]
    if not any(key):
        return 0
    stripped = [ln.strip() for ln in lines]
    n = len(key)
    hits = [i for i in range(len(lines) - n + 1) if stripped[i : i + n] == key]
    if len(hits) != 1:
        return len(hits)
    i = hits[0]
    new_lines = new.strip("\n").split("\n") if new.strip("\n") else []
    file_base, old_base = _base_indent(lines[i : i + n]), _base_indent(old_lines)
    # The model copied the block shifted (usually without its indentation): shift
    # the new text the same way. If old and new disagree, new_text is trusted as is.
    if file_base != old_base and _base_indent(new_lines) == old_base:
        new_lines = [file_base + ln[len(old_base):] if ln.strip() else ln for ln in new_lines]
    return "\n".join(lines[:i] + new_lines + lines[i + n :]), i, len(new_lines)


def _closest_match(content: str, old: str) -> str:
    """The region of *content* most like *old*, if one is close."""
    lines = content.split("\n")
    n = len(old.strip("\n").split("\n"))
    if len(lines) > 5000 or n > 200:
        return ""
    target = "\n".join(ln.strip() for ln in old.strip("\n").split("\n"))
    best, best_i = 0.0, -1
    for i in range(max(1, len(lines) - n + 1)):
        m = difflib.SequenceMatcher(None, "\n".join(ln.strip() for ln in lines[i : i + n]), target)
        if m.real_quick_ratio() > best and m.quick_ratio() > best and (r := m.ratio()) > best:
            best, best_i = r, i
    if best < 0.6:
        return ""
    return f"\nClosest match (line {best_i + 1}, {best:.0%} similar):\n" + "\n".join(lines[best_i : best_i + n])


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill *proc* and every child it spawned (it runs in its own session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _communicate(proc: asyncio.subprocess.Process, timeout: float) -> tuple[bytes, bytes] | None:
    """Wait for *proc*; None if it timed out. Killed on timeout and when the turn is cancelled (Ctrl+C)."""
    try:
        return await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        _kill(proc)
        await proc.wait()
        return None
    except BaseException:
        _kill(proc)
        raise


# ── The registry ─────────────────────────────────────────────────────


class Tools:
    """The tools available under a config, and how to run them."""

    def __init__(self, config: dict, memory: Memory):
        cfg = with_defaults(config)
        tools_cfg = cfg["tools"]
        self.workspace = Path(cfg["workspace"])
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sandbox_level: str = tools_cfg["sandboxLevel"]
        self._memory = memory
        self._shell_timeout = float(tools_cfg["shellTimeout"])
        allow = set(tools_cfg["shellEnvAllow"])
        self._shell_env = {k: v for k, v in os.environ.items() if k in allow or not _SECRET_ENV.search(k)}
        self._blocked = [(re.compile(p, re.IGNORECASE), _BLOCKED_COMMANDS.get(p, "custom rule"))
                         for p in [*_BLOCKED_COMMANDS, *tools_cfg["blockedCommands"]]]
        self._http = httpx.AsyncClient(timeout=30)
        # (path, original bytes or None if the file did not exist), per file change.
        self._undo: list[tuple[Path, bytes | None]] = []
        # Files read or written here. Others are not overwritten blind: small
        # models "append" by writing only the new line.
        self._seen: set[Path] = set()
        self.approver: Approver | None = None

        readonly = self.sandbox_level == "readonly"
        self._tools: dict[str, Tool] = {}
        self._unavailable: dict[str, str] = {}  # name -> why it is missing
        for attr in vars(type(self)).values():
            if not hasattr(attr, "__tool__"):
                continue
            description, writes, docs = attr.__tool__
            method = getattr(self, attr.__name__)
            accepted = set(inspect.signature(method).parameters)
            t = Tool(_schema(attr, description, docs),
                     lambda args, m=method, ok=accepted: m(**{k: v for k, v in args.items() if k in ok}))
            if readonly and writes:
                self._unavailable[t.name] = "the workspace is read-only"
            elif t.name in ("remember", "recall") and not cfg["memory"]["enabled"]:
                self._unavailable[t.name] = "memory is turned off"
            else:
                self._tools[t.name] = t
        self._load_plugins(readonly)

        self.confirm = {TOOL_ALIASES.get(n, n) for n in tools_cfg["confirm"]}
        unknown = self.confirm - set(self._tools) - set(self._unavailable)
        if unknown:
            raise ValueError(f"Unknown tool(s) in tools.confirm: {', '.join(sorted(unknown))}")

    def _load_plugins(self, readonly: bool) -> None:
        """Each ``<home>/plugins/*.py`` exporting ``TOOL_DEF`` and ``handler`` becomes a tool."""
        for path in sorted((agent_home() / "plugins").glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(f"agent_mini_plugin_{path.stem}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                log.warning("Failed to load plugin %s: %s", path, e)
                continue
            definition, handler = getattr(module, "TOOL_DEF", None), getattr(module, "handler", None)
            func = definition.get("function") if isinstance(definition, dict) else None
            name = func.get("name") if isinstance(func, dict) else None
            if not (isinstance(name, str) and name and callable(handler)):
                log.warning("Skipping plugin %s: it needs TOOL_DEF with function.name, and a handler", path)
                continue
            if name in self._tools or name in self._unavailable or name in TOOL_ALIASES:
                log.warning("Skipping plugin %s: '%s' is a built-in tool name", path, name)
            elif readonly and not definition.get("x-readonly"):
                self._unavailable[name] = "the workspace is read-only (plugin lacks x-readonly)"
            else:
                # Providers reject unknown keys in tool definitions.
                clean = {k: v for k, v in definition.items() if k != "x-readonly"}
                self._tools[name] = Tool(clean, handler)
                log.info("Loaded plugin: %s from %s", name, path)

    def definitions(self) -> list[dict]:
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def known_names(self) -> set[str]:
        """Names a model may use for a tool, aliases included (for recovering calls written as text)."""
        return set(self._tools) | {a for a, n in TOOL_ALIASES.items() if n in self._tools}

    async def close(self) -> None:
        await self._http.aclose()

    async def execute(self, name: str, arguments: dict) -> str:
        """Run tool *name*. Every failure comes back as an ``Error:`` string the model can act on."""
        name = name if isinstance(name, str) else ""
        name = name if name in self._tools else TOOL_ALIASES.get(name, name)
        t = self._tools.get(name)
        if t is None:
            if name in self._unavailable:
                return f"Error: {name} is not available: {self._unavailable[name]}."
            if name and "shell" in self._tools and shutil.which(name):  # "python", "git", "cat", …
                return f"Error: there is no tool '{name}'. To run the {name} program, call shell with the command."
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}"
        if INVALID_ARGS_KEY in arguments:
            return (f"Error: the arguments for {name} were not valid JSON: "
                    f"{str(arguments[INVALID_ARGS_KEY])[:200]}. Send the call again with valid JSON.")
        args = _normalize(t.params, arguments)
        required = t.params.get("required", [])
        if missing := [r for r in required if r not in args]:
            props = t.params.get("properties", {})
            expected = ", ".join(f'"{r}": <{props.get(r, {}).get("type", "string")}>' for r in required)
            return f"Error: missing {', '.join(missing)} for {name}. Expected {{{expected}}}"
        if name in self.confirm:
            if self.approver is None:
                return (f"Error: {name} needs the user's approval, and nobody can approve it here. "
                        "Tell the user what you wanted to do instead.")
            if not await self.approver(name, args):
                return f"Error: the user declined this {name} call. Don't retry it; ask how to proceed."
        try:
            result = t.call(args)
            return str(await result if inspect.isawaitable(result) else result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # ── Paths, undo and previews ─────────────────────────────────────

    def _resolve(self, raw: str) -> Path:
        ws = self.workspace.resolve()
        given = Path(raw).expanduser()
        p = (given if given.is_absolute() else ws / given).resolve()
        if self.sandbox_level == "unrestricted" or p.is_relative_to(ws):
            return p
        # Small models write "/notes.txt" or "/workspace/notes.txt" for a file in the
        # workspace. When the top folder doesn't exist outside, that is what they mean.
        parts = given.parts[1:] if given.is_absolute() else ()
        if given.is_absolute() and not parts:
            return ws  # "/" itself
        if parts and not Path(given.anchor, parts[0]).exists():
            inside = (ws.joinpath(*parts[1:]) if parts[0] == "workspace" else ws.joinpath(*parts)).resolve()
            if inside.is_relative_to(ws):
                return inside
        raise PermissionError(f"Access denied: {p} is outside the workspace. Use a relative path like 'notes.txt'.")

    def _show(self, p: Path) -> str:
        """Workspace-relative path when possible (saves tokens)."""
        ws = self.workspace.resolve()
        return str(p.relative_to(ws)) if p.is_relative_to(ws) else str(p)

    def _checkpoint(self, p: Path) -> None:
        self._undo.append((p, p.read_bytes() if p.exists() else None))
        del self._undo[:-_UNDO_LIMIT]

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    def undo(self) -> str:
        """Revert the latest file change made by write_file or edit_file."""
        if not self._undo:
            return "Nothing to undo."
        p, original = self._undo.pop()
        if original is None:
            p.unlink(missing_ok=True)
            return f"Deleted {self._show(p)} (the agent created it)."
        p.write_bytes(original)
        return f"Restored {self._show(p)}."

    def preview(self, name: str, arguments: dict) -> str | None:
        """Unified diff of what a write_file/edit_file call would change; None for other tools."""
        name = TOOL_ALIASES.get(name, name)
        if name not in ("write_file", "edit_file"):
            return None
        args = _normalize(self._tools[name].params, arguments) if name in self._tools else arguments
        try:
            p = self._resolve(args.get("path", ""))
            old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if name == "write_file":
                new = args.get("content", "")
            else:
                planned = self._plan_edit(p, args.get("old_text", ""), args.get("new_text", ""))
                if isinstance(planned, str):
                    return planned
                new = planned[0]
        except Exception as e:
            return f"(no preview: {e})"
        shown = self._show(p)
        diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{shown}", f"b/{shown}")
        return "".join(diff) or "(no changes)"

    # ── Tools ────────────────────────────────────────────────────────

    @tool("Run a shell command in the workspace. Returns its output, and the exit code if it failed.",
          writes=True, command="The command, e.g. `ls -la` or `python -m pytest -q`.")
    async def shell(self, command: str) -> str:
        for pattern, rule in self._blocked:
            if pattern.search(command):
                return (f"Error: command blocked by security policy ({rule}). "
                        "If this is intentional, adjust tools.blockedCommands in the config.")
        proc = await asyncio.create_subprocess_shell(
            command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(self.workspace), env=self._shell_env, start_new_session=True,
        )
        done = await _communicate(proc, self._shell_timeout)
        if done is None:
            return f"Error: command timed out after {self._shell_timeout:g} seconds"
        out = "\n".join(s.decode(errors="replace").rstrip("\n") for s in done if s) or "(no output)"
        # No "Error:" prefix: grep exiting 1 on no match is not a failure.
        return out + (f"\n[exit code: {proc.returncode}]" if proc.returncode else "")

    @tool("Read a text file. For a big file, read part of it with offset and limit.",
          path="File path.", offset="First line to read, starting at 1; negative counts from the end. Optional.",
          limit="How many lines to read. Optional.")
    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> str:
        p = self._resolve(path)
        if not p.is_file():
            what = "is a directory; use list_files" if p.is_dir() else "does not exist"
            return f"Error: {self._show(p)} {what}."
        with open(p, "rb") as f:
            if b"\0" in f.read(8192):
                return f"Error: {self._show(p)} is a binary file ({p.stat().st_size} bytes)."
        content = p.read_text(encoding="utf-8", errors="replace")
        self._seen.add(p)
        if not (offset or limit):
            return content or "(empty file)"
        lines = content.splitlines(keepends=True)
        start = int(offset or 1)
        start = max(1, len(lines) + start + 1 if start < 0 else start)  # -50: the last 50 lines
        if lines and start > len(lines):
            return f"Error: offset {start} is past the end of the file ({len(lines)} lines)."
        end = min(len(lines), start - 1 + int(limit)) if limit else len(lines)
        return f"[lines {start}-{end} of {len(lines)}]\n" + "".join(lines[start - 1 : end])

    @tool("Create a file, or replace a file's whole content.", writes=True,
          path="File path.", content="The complete file content.")
    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p.exists() and p not in self._seen:
            return (f"Error: {self._show(p)} already exists. Read it first; then use edit_file to change it, "
                    "or write_file again to replace all of it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint(p)
        p.write_text(content, encoding="utf-8")
        self._seen.add(p)
        n = len(content.splitlines())
        return f"Wrote {self._show(p)} ({n} line{'' if n == 1 else 's'})."

    def _plan_edit(self, p: Path, old_text: str, new_text: str) -> str | tuple[str, int, int, bool, bool]:
        """Work out an edit without writing it: an ``Error:`` string, or
        ``(new_content, first_line, new_line_count, fuzzy, crlf)`` with ``\\n`` line ends."""
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: {self._show(p)} is not valid UTF-8; refusing to edit it."
        crlf = "\r\n" in raw
        content, old, new = (s.replace("\r\n", "\n") for s in (raw, old_text, new_text))
        count = content.count(old)
        if count == 1:
            start = content.index(old)
            return content[:start] + new + content[start + len(old):], content.count("\n", 0, start), \
                new.count("\n") + 1, False, crlf
        match = _fuzzy_replace(content, old, new) if count == 0 else count
        if isinstance(match, int):
            if match > 1:
                return (f"Error: old_text matches {match} places. "
                        "Include more surrounding lines so it matches exactly once.")
            return ("Error: old_text was not found. Copy it exactly from read_file output, "
                    "including indentation." + _closest_match(content, old))
        return (*match, True, crlf)

    @tool("Replace text in a file. old_text must match exactly one place; "
          "include a few surrounding lines to make it unique.", writes=True,
          path="File path.", old_text="The exact text to replace, copied from read_file output.",
          new_text="The text to put in its place.")
    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self._resolve(path)
        shown = self._show(p)
        if not p.is_file():
            return f"Error: {shown} does not exist. Use write_file to create it."
        if not old_text:
            return "Error: old_text is empty. Use write_file to create or rewrite a file."
        planned = self._plan_edit(p, old_text, new_text)
        if isinstance(planned, str):
            return planned
        content, first, n_new, fuzzy, crlf = planned
        self._checkpoint(p)
        p.write_bytes((content.replace("\n", "\r\n") if crlf else content).encode("utf-8"))
        self._seen.add(p)
        # Showing the result saves the model a read_file call to check it.
        lines = content.rstrip("\n").split("\n")
        lo, hi = max(0, first - 2), min(len(lines), first + max(n_new, 1) + 2)
        summary = f"Edited {shown}" + (" (matched ignoring whitespace)" if fuzzy else "") + "."
        if hi - lo > 30:
            return summary
        return f"{summary} Lines {lo + 1}-{hi} now:\n" + "\n".join(lines[lo:hi])

    @tool("List a directory, or find files recursively by glob pattern (e.g. '*.py', 'src/**/test_*.py').",
          path="Directory. Default: the workspace.", pattern="Glob pattern. Leave out to list the directory.")
    def list_files(self, path: str = ".", pattern: str = "") -> str:
        root = self._resolve(path)
        if not root.is_dir():
            return f"Error: {self._show(root)} is not a directory."
        if not pattern.strip():
            entries = sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:200]]
            if len(entries) > 200:
                lines.append(f"… and {len(entries) - 200} more entries")
            return "\n".join(lines) or "(empty directory)"
        pattern = pattern.strip().removeprefix("./")
        # "*.py" matches at any depth, and "**/" also matches no folder at all
        # ("src/**/*.py" includes src/a.py).
        name_only, flat = "/" not in pattern, pattern.replace("**/", "")
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(root)
            for fn in sorted(filenames):
                rel = (rel_dir / fn).as_posix()
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, flat) or \
                        (name_only and fnmatch.fnmatch(fn, pattern)):
                    found.append(self._show(Path(dirpath) / fn))
            if len(found) > 10_000:
                break
        if not found:
            return f"No files matching '{pattern}'."
        found.sort()
        return "\n".join(found[:200]) + (f"\n… and {len(found) - 200} more" if len(found) > 200 else "")

    @tool("Search file contents for a regex, like grep. Returns path:line:text for each match.",
          query="The regex to search for.", path="Directory or file to search. Default: the workspace.")
    async def search_files(self, query: str, path: str = ".") -> str:
        target = self._show(self._resolve(path))
        # "-e" and "--" stop a query that starts with "-" from being read as a flag.
        if rg := shutil.which("rg"):
            cmd = [rg, "--line-number", "--no-heading", "--hidden", "--max-columns", "300",
                   *[a for d in _SKIP_DIRS for a in ("--glob", f"!{d}")], "-e", query, "--", target]
        else:
            cmd = [shutil.which("grep") or "grep", "-rnI", *[f"--exclude-dir={d}" for d in _SKIP_DIRS],
                   "-e", query, "--", target]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace.resolve()), start_new_session=True,
        )
        done = await _communicate(proc, 30)
        if done is None:
            return "Error: search timed out after 30 seconds. Narrow the path or pattern."
        if proc.returncode not in (0, 1):
            return f"Error: search failed: {done[1].decode(errors='replace').strip() or 'unknown error'}"
        out = re.sub(r"(?m)^\./", "", done[0].decode(errors="replace").strip())  # "./a.py:1:…" → "a.py:1:…"
        return out or "No matches found."

    @tool("Search the web. Returns titles, URLs and snippets.", query="What to search for.")
    async def web_search(self, query: str) -> str:
        return await web.web_search(self._http, query)

    @tool("Fetch a web page as plain text.", url="An http(s) URL.")
    async def web_fetch(self, url: str) -> str:
        return await web.web_fetch(self._http, url)

    @tool("Save a fact for later sessions, e.g. a preference of the user. Saving the same key again updates it.",
          key="Short label, e.g. 'editor'.", value="The fact.")
    def remember(self, key: str, value: str) -> str:
        return self._memory.store(key, value)

    @tool("Search saved facts.", query="Keywords.")
    def recall(self, query: str) -> str:
        return self._memory.recall(query)
