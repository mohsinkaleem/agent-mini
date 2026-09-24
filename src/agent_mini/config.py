"""Configuration loading and management."""

from __future__ import annotations

import json
import os
from pathlib import Path


def agent_home() -> Path:
    """Root for config, memory, sessions and plugins. ``AGENT_MINI_HOME`` overrides it."""
    return Path(os.environ.get("AGENT_MINI_HOME") or Path.home() / ".agent-mini").expanduser()


def write_private(path: Path, text: str) -> None:
    """Atomically write *text* to *path* with owner-only (0600) permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


CONFIG_DIR = agent_home()
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_WORKSPACE = CONFIG_DIR / "workspace"
MEMORY_FILE = CONFIG_DIR / "memory.json"


def load_config() -> dict:
    """Load config from ~/.agent-mini/config.json."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Save config to ~/.agent-mini/config.json (0600: it can hold API keys)."""
    write_private(CONFIG_FILE, json.dumps(config, indent=2))
