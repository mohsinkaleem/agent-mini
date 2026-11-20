"""The tools the model can call, behind one registry.

Defining a tool: a ``Tools`` method marked with ``@tool``. Its signature becomes
the JSON schema the model sees (see ``_schema``), so the code and the schema
can't drift apart.

Running a tool: ``Tools.execute(name, arguments)`` is the only entry point. It
never raises: every failure comes back as an ``Error: …`` string that the model
reads and can act on.

Helpers live next door: ``sandbox.py`` (paths, blocked commands, processes),
``editing.py`` (how edit_file finds its text) and ``forgiving.py`` (the names
and argument shapes small models use instead of ours).
"""

from __future__ import annotations

import difflib
import fnmatch
import importlib.util
import inspect
import logging
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import web
from .config import agent_home, with_defaults
from .editing import EditPlan, plan_edit
from .forgiving import TOOL_ALIASES, normalize_arguments
from .memory import Memory
from .providers.parsing import INVALID_ARGS_KEY
from .sandbox import blocked_reason, compile_blocklist, resolve_path, run_process, shell_env

log = logging.getLogger("agent-mini")

# Asked before running a tool listed in tools.confirm: (name, arguments) -> allowed?
Approver = Callable[[str, dict], Awaitable[bool]]

# Directories that are never worth listing recursively or searching.
_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
                        ".mypy_cache", ".ruff_cache", ".tox"})
_UNDO_LIMIT = 50  # file changes /undo can revert
_LIST_LIMIT = 200  # entries list_files shows
_GLOB_SCAN_LIMIT = 10_000  # matches after which list_files stops walking
_SEARCH_TIMEOUT = 30  # seconds
_EDIT_ECHO_MAX_LINES = 30  # edit_file shows the edited lines when there are at most this many


# ── Defining tools ───────────────────────────────────────────────────


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
    """The JSON schema for a tool method: parameters without a default are required."""
    props, required = {}, []
    for p in list(inspect.signature(fn).parameters.values())[1:]:  # skip self
        kind = "integer" if "int" in str(p.annotation) else "string"
        props[p.name] = {"type": kind, "description": docs.get(p.name, "")}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


def _with_known_args(method: Callable) -> Callable[[dict], str | Awaitable[str]]:
    """A function that calls *method* with only the arguments it accepts."""
    accepted = set(inspect.signature(method).parameters)

    def call(args: dict):
        return method(**{k: v for k, v in args.items() if k in accepted})
    return call


def _glob_match(rel_path: str, pattern: str) -> bool:
    """Whether a workspace-relative path matches a glob the way people (and models) expect it to.

    "*.py" matches at any depth, and "**/" also matches no folder at all
    ("src/**/*.py" includes src/a.py).
    """
    if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, pattern.replace("**/", "")):
        return True
    file_name = rel_path.rsplit("/", 1)[-1]
    return "/" not in pattern and fnmatch.fnmatch(file_name, pattern)


# ── The registry ─────────────────────────────────────────────────────


class Tools:
    """The tools available under a config, and how to run them."""

    def __init__(self, config: dict, memory: Memory):
        cfg = with_defaults(config)
        tools_cfg = cfg["tools"]
        self.workspace = Path(cfg["workspace"])
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sandbox_level: str = tools_cfg["sandboxLevel"]
        self.approver: Approver | None = None  # set by the CLI; None means nobody can approve

        self._memory = memory
        self._http = httpx.AsyncClient(timeout=30)
        self._shell_timeout = float(tools_cfg["shellTimeout"])
        self._shell_env = shell_env(tools_cfg["shellEnvAllow"])
        self._blocklist = compile_blocklist(tools_cfg["blockedCommands"])
        # (path, original bytes or None if the file did not exist), one per file change.
        self._undo: list[tuple[Path, bytes | None]] = []
        # Files read or written in this session (see write_file).
        self._seen: set[Path] = set()

        readonly = self.sandbox_level == "readonly"
        self._tools: dict[str, Tool] = {}
        self._unavailable: dict[str, str] = {}  # name -> why it is missing
        self._register_builtins(readonly, memory_on=cfg["memory"]["enabled"])
        self._load_plugins(readonly)

        self.confirm = {TOOL_ALIASES.get(n, n) for n in tools_cfg["confirm"]}
        unknown = self.confirm - set(self._tools) - set(self._unavailable)
        if unknown:  # a typo must not silently turn approval off
            raise ValueError(f"Unknown tool(s) in tools.confirm: {', '.join(sorted(unknown))}")

    def _register_builtins(self, readonly: bool, memory_on: bool) -> None:
        """Every method marked with ``@tool`` becomes a tool, unless the config leaves it out."""
        for attr in vars(type(self)).values():
            if not hasattr(attr, "__tool__"):
                continue
            description, writes, docs = attr.__tool__
            t = Tool(_schema(attr, description, docs), _with_known_args(getattr(self, attr.__name__)))
            if readonly and writes:
                self._unavailable[t.name] = "the workspace is read-only"
            elif t.name in ("remember", "recall") and not memory_on:
                self._unavailable[t.name] = "memory is turned off"
            else:
                self._tools[t.name] = t

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
            definition = getattr(module, "TOOL_DEF", None)
            handler = getattr(module, "handler", None)
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
        return set(self._tools) | {alias for alias, name in TOOL_ALIASES.items() if name in self._tools}

    async def close(self) -> None:
        await self._http.aclose()

    # ── Running a tool ───────────────────────────────────────────────

    async def execute(self, name: str, arguments: dict) -> str:
        """Run tool *name*. Every failure comes back as an ``Error:`` string the model can act on."""
        # 1. Find the tool, accepting the aliases models use.
        name = name if isinstance(name, str) else ""
        if name not in self._tools:
            name = TOOL_ALIASES.get(name, name)
        t = self._tools.get(name)
        if t is None:
            return self._unknown_tool_error(name)

        # 2. Check and normalize the arguments.
        if INVALID_ARGS_KEY in arguments:
            return (f"Error: the arguments for {name} were not valid JSON: "
                    f"{str(arguments[INVALID_ARGS_KEY])[:200]}. Send the call again with valid JSON.")
        args = normalize_arguments(t.params, arguments)
        required = t.params.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            props = t.params.get("properties", {})
            expected = ", ".join(f'"{r}": <{props.get(r, {}).get("type", "string")}>' for r in required)
            return f"Error: missing {', '.join(missing)} for {name}. Expected {{{expected}}}"

        # 3. Ask the user, if the config says this tool needs approval.
        if name in self.confirm:
            if self.approver is None:
                return (f"Error: {name} needs the user's approval, and nobody can approve it here. "
                        "Tell the user what you wanted to do instead.")
            if not await self.approver(name, args):
                return f"Error: the user declined this {name} call. Don't retry it; ask how to proceed."

        # 4. Run it. Tools may be plain functions or coroutines.
        try:
            result = t.call(args)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _unknown_tool_error(self, name: str) -> str:
        if name in self._unavailable:
            return f"Error: {name} is not available: {self._unavailable[name]}."
        # Small-model fix: models call programs as if they were tools ("python", "git", "cat").
        if name and "shell" in self._tools and shutil.which(name):
            return f"Error: there is no tool '{name}'. To run the {name} program, call shell with the command."
        return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}"

    # ── Paths, undo and previews ─────────────────────────────────────

    def _resolve(self, raw: str) -> Path:
        return resolve_path(raw, self.workspace, self.sandbox_level)

    def _show(self, p: Path) -> str:
        """Workspace-relative path when possible (saves tokens)."""
        ws = self.workspace.resolve()
        return str(p.relative_to(ws)) if p.is_relative_to(ws) else str(p)

    def _checkpoint(self, p: Path) -> None:
        """Remember a file's current bytes so /undo can restore them."""
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
        args = normalize_arguments(self._tools[name].params, arguments) if name in self._tools else arguments
        try:
            p = self._resolve(args.get("path", ""))
            old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if name == "write_file":
                new = args.get("content", "")
            else:
                planned = self._plan_edit(p, args.get("old_text", ""), args.get("new_text", ""))
                if isinstance(planned, str):
                    return planned
                new = planned.content
        except Exception as e:
            return f"(no preview: {e})"
        shown = self._show(p)
        diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{shown}", f"b/{shown}")
        return "".join(diff) or "(no changes)"

    def _plan_edit(self, p: Path, old_text: str, new_text: str) -> EditPlan | str:
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: {self._show(p)} is not valid UTF-8; refusing to edit it."
        return plan_edit(raw, old_text, new_text)

    # ── The tools ────────────────────────────────────────────────────

    @tool("Run a shell command in the workspace. Returns its output, and the exit code if it failed.",
          writes=True, command="The command, e.g. `ls -la` or `python -m pytest -q`.")
    async def shell(self, command: str) -> str:
        if reason := blocked_reason(command, self._blocklist):
            return (f"Error: command blocked by security policy ({reason}). "
                    "If this is intentional, adjust tools.blockedCommands in the config.")
        done = await run_process(command, cwd=str(self.workspace), timeout=self._shell_timeout, env=self._shell_env)
        if done is None:
            return f"Error: command timed out after {self._shell_timeout:g} seconds"
        code, stdout, stderr = done
        parts = [s.decode(errors="replace").rstrip("\n") for s in (stdout, stderr) if s]
        output = "\n".join(parts) or "(no output)"
        # No "Error:" prefix: grep exiting 1 on no match is not a failure.
        if code:
            output += f"\n[exit code: {code}]"
        return output

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
        if start < 0:  # Small-model fix: -50 means the last 50 lines, like `tail -n 50`.
            start = len(lines) + start + 1
        start = max(1, start)
        if lines and start > len(lines):
            return f"Error: offset {start} is past the end of the file ({len(lines)} lines)."
        end = min(len(lines), start - 1 + int(limit)) if limit else len(lines)
        return f"[lines {start}-{end} of {len(lines)}]\n" + "".join(lines[start - 1 : end])

    @tool("Create a file, or replace a file's whole content.", writes=True,
          path="File path.", content="The complete file content.")
    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        # Small-model fix: no blind overwrites. Small models "append" a line by
        # writing only that line, so a file must be read before it is replaced.
        if p.exists() and p not in self._seen:
            return (f"Error: {self._show(p)} already exists. Read it first; then use edit_file to change it, "
                    "or write_file again to replace all of it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint(p)
        p.write_text(content, encoding="utf-8")
        self._seen.add(p)
        n = len(content.splitlines())
        return f"Wrote {self._show(p)} ({n} line{'' if n == 1 else 's'})."

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
        plan = self._plan_edit(p, old_text, new_text)
        if isinstance(plan, str):
            return plan
        self._checkpoint(p)
        content = plan.content.replace("\n", "\r\n") if plan.crlf else plan.content
        p.write_bytes(content.encode("utf-8"))
        self._seen.add(p)

        summary = f"Edited {shown}" + (" (matched ignoring whitespace)" if plan.fuzzy else "") + "."
        # Showing the edited lines (plus two either side) saves the model a read_file to check them.
        lines = plan.content.rstrip("\n").split("\n")
        top = max(0, plan.first_line - 2)
        bottom = min(len(lines), plan.first_line + max(plan.new_line_count, 1) + 2)
        if bottom - top > _EDIT_ECHO_MAX_LINES:
            return summary
        return f"{summary} Lines {top + 1}-{bottom} now:\n" + "\n".join(lines[top:bottom])

    @tool("List a directory, or find files recursively by glob pattern (e.g. '*.py', 'src/**/test_*.py').",
          path="Directory. Default: the workspace.", pattern="Glob pattern. Leave out to list the directory.")
    def list_files(self, path: str = ".", pattern: str = "") -> str:
        root = self._resolve(path)
        if not root.is_dir():
            return f"Error: {self._show(root)} is not a directory."

        if not pattern.strip():
            entries = sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name))  # folders first
            lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:_LIST_LIMIT]]
            if len(entries) > _LIST_LIMIT:
                lines.append(f"… and {len(entries) - _LIST_LIMIT} more entries")
            return "\n".join(lines) or "(empty directory)"

        pattern = pattern.strip().removeprefix("./")
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(root)
            for file_name in sorted(filenames):
                if _glob_match((rel_dir / file_name).as_posix(), pattern):
                    found.append(self._show(Path(dirpath) / file_name))
            if len(found) > _GLOB_SCAN_LIMIT:
                break
        if not found:
            return f"No files matching '{pattern}'."
        found.sort()
        more = f"\n… and {len(found) - _LIST_LIMIT} more" if len(found) > _LIST_LIMIT else ""
        return "\n".join(found[:_LIST_LIMIT]) + more

    @tool("Search file contents for a regex, like grep. Returns path:line:text for each match.",
          query="The regex to search for.", path="Directory or file to search. Default: the workspace.")
    async def search_files(self, query: str, path: str = ".") -> str:
        target = self._show(self._resolve(path))
        if rg := shutil.which("rg"):
            cmd = [rg, "--line-number", "--no-heading", "--hidden", "--max-columns", "300"]
            for d in _SKIP_DIRS:
                cmd += ["--glob", f"!{d}"]
        else:
            cmd = [shutil.which("grep") or "grep", "-rnI", *[f"--exclude-dir={d}" for d in _SKIP_DIRS]]
        # "-e" and "--" stop a query that starts with "-" from being read as a flag.
        cmd += ["-e", query, "--", target]

        done = await run_process(cmd, cwd=str(self.workspace.resolve()), timeout=_SEARCH_TIMEOUT)
        if done is None:
            return f"Error: search timed out after {_SEARCH_TIMEOUT} seconds. Narrow the path or pattern."
        code, stdout, stderr = done
        if code not in (0, 1):  # 1 means no matches
            return f"Error: search failed: {stderr.decode(errors='replace').strip() or 'unknown error'}"
        out = re.sub(r"(?m)^\./", "", stdout.decode(errors="replace").strip())  # "./a.py:1:…" → "a.py:1:…"
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
