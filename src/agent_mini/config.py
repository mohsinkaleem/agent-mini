"""Configuration loading and management."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".agent-mini"
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
    """Save config to ~/.agent-mini/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_workspace(config: dict | None = None) -> Path:
    """Get workspace directory path."""
    if config is None:
        config = load_config()
    ws = Path(config.get("workspace", str(DEFAULT_WORKSPACE))).expanduser()
    ws.mkdir(parents=True, exist_ok=True)
    return ws
