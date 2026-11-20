"""Keeping the agent in its box.

Which paths it may touch, which commands it may run, which environment
variables it sees, and processes that can always be killed.

These are guard rails, not a security boundary: asking the user before a tool
runs (``tools.confirm``) is the real control. See SECURITY.md.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from collections.abc import Iterable
from pathlib import Path

# Env vars whose names look like secrets are not passed to the shell.
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)

# Commands refused outright: pattern -> the reason shown to the model.
BLOCKED_COMMANDS = {
    # rm with both a recursive and a force flag, in any order or spelling.
    r"\brm\b(?=[^|;&\n]*\s(?:-[a-z]*r|--recursive\b))(?=[^|;&\n]*\s(?:-[a-z]*f|--force\b))": "destructive rm -rf",
    r"\bsudo\b": "sudo elevation",
    r"\bmkfs\b": "filesystem format",
    r"\bdd\s+if=": "raw disk write",
    r":\(\)\s*\{": "fork bomb",
    r"\b(chmod|chown)\s+(-R\s+)?[0-7]*\s+/[^\s]*$": "root permission change",
}

Blocklist = list[tuple[re.Pattern, str]]


# ── Environment and commands ─────────────────────────────────────────


def shell_env(allow: Iterable[str]) -> dict[str, str]:
    """The environment for shell commands: ours, minus variables that look like secrets."""
    allow = set(allow)
    return {name: value for name, value in os.environ.items() if name in allow or not SECRET_ENV.search(name)}


def compile_blocklist(extra_patterns: Iterable[str]) -> Blocklist:
    """The built-in blocked-command patterns plus the user's, each with its reason."""
    patterns = [*BLOCKED_COMMANDS, *extra_patterns]
    return [(re.compile(p, re.IGNORECASE), BLOCKED_COMMANDS.get(p, "custom rule")) for p in patterns]


def blocked_reason(command: str, blocklist: Blocklist) -> str | None:
    """Why *command* is refused, or None if it may run."""
    for pattern, reason in blocklist:
        if pattern.search(command):
            return reason
    return None


# ── Paths ────────────────────────────────────────────────────────────


def resolve_path(raw: str, workspace: Path, level: str) -> Path:
    """The real path for *raw*. Raises PermissionError if the sandbox *level* doesn't allow it.

    Relative paths are relative to the workspace. In the ``workspace`` and
    ``readonly`` levels the result must stay inside the workspace.
    """
    ws = workspace.resolve()
    given = Path(raw).expanduser()
    path = (given if given.is_absolute() else ws / given).resolve()
    if level == "unrestricted" or path.is_relative_to(ws):
        return path

    # Small-model fix: models write "/notes.txt" or "/workspace/notes.txt" for a
    # file in the workspace. When the top folder doesn't exist on the real
    # filesystem, that is what they mean. A real folder (/etc) is still refused.
    if given.is_absolute():
        parts = given.parts[1:]
        if not parts:
            return ws  # "/" itself
        if not Path(given.anchor, parts[0]).exists():
            if parts[0] == "workspace":
                parts = parts[1:]
            inside = ws.joinpath(*parts).resolve()
            if inside.is_relative_to(ws):
                return inside
    raise PermissionError(f"Access denied: {path} is outside the workspace. Use a relative path like 'notes.txt'.")


# ── Processes ────────────────────────────────────────────────────────


async def run_process(
    command: str | list[str], cwd: str, timeout: float, env: dict[str, str] | None = None
) -> tuple[int, bytes, bytes] | None:
    """Run *command* (a shell string, or a program and its arguments) and wait for it.

    Returns ``(exit_code, stdout, stderr)``, or None if it timed out.

    stdin is closed, so a command that asks for input gets EOF instead of hanging.
    It runs in its own session, so on a timeout, or when the turn is cancelled
    (Ctrl+C), the whole process group is killed, children included.
    """
    options = dict(stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                   stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env, start_new_session=True)
    if isinstance(command, str):
        proc = await asyncio.create_subprocess_shell(command, **options)
    else:
        proc = await asyncio.create_subprocess_exec(*command, **options)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        _kill(proc)
        await proc.wait()
        return None
    except BaseException:  # cancelled: don't leave the process running
        _kill(proc)
        raise
    return proc.returncode, stdout, stderr


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill *proc* and every child it spawned (it runs in its own session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
