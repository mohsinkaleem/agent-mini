"""Configuration: paths, defaults, loading and validation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

PROVIDERS = ("ollama", "openai", "local")
SANDBOX_LEVELS = ("unrestricted", "workspace", "readonly")
TIERS = ("tiny", "small", "medium", "large", "cloud")

# Every setting and its default. config.example.json is this, written out.
DEFAULTS: dict = {
    "provider": "ollama",
    "providers": {
        "ollama": {"baseUrl": "http://localhost:11434", "model": "qwen3.5:9b", "think": False,
                   "numCtx": None, "keepAlive": None},
        "openai": {"baseUrl": "https://api.openai.com/v1", "apiKey": "", "model": "gpt-5-mini",
                   "reasoningEffort": None},
        "local": {"baseUrl": "http://localhost:8080/v1", "apiKey": "", "model": "local-model",
                  "reasoningEffort": None},
    },
    "agent": {
        "tier": None,           # tiny/small/medium/large/cloud; null = from the model size
        "contextWindow": None,  # token budget; null = the tier's
        "maxIterations": None,  # null = the tier's
        "temperature": None,    # null = the tier's
        "systemPrompt": "",
        "showThinking": False,
    },
    "tools": {
        "sandboxLevel": "workspace",
        "confirm": [],
        "shellTimeout": 120,
        "shellEnvAllow": [],
        "blockedCommands": [],
    },
    "memory": {"enabled": True, "maxEntries": 1000},
    "channels": {
        "telegram": {"enabled": False, "token": "", "allowFrom": [], "allowShell": False,
                     "sharedMemory": False, "streamResponses": True},
    },
    "workspace": None,  # null = <home>/workspace
}


def agent_home() -> Path:
    """Root for config, memory, sessions and plugins. ``AGENT_MINI_HOME`` overrides it."""
    return Path(os.environ.get("AGENT_MINI_HOME") or Path.home() / ".agent-mini").expanduser()


def config_file() -> Path:
    return agent_home() / "config.json"


def write_private(path: Path, text: str) -> None:
    """Atomically write *text* to *path* with owner-only (0600) permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def with_defaults(config: dict) -> dict:
    """*config* layered over DEFAULTS, validated. Raises ValueError on a bad value."""
    cfg = _merge(DEFAULTS, config)
    if cfg["provider"] not in PROVIDERS:
        raise ValueError(f"Unknown provider {cfg['provider']!r}. Use one of: {', '.join(PROVIDERS)}.")
    level = str(cfg["tools"]["sandboxLevel"]).strip().lower()
    if level not in SANDBOX_LEVELS:
        # A typo must not quietly mean "no sandbox".
        raise ValueError(
            f"Invalid tools.sandboxLevel {cfg['tools']['sandboxLevel']!r}. Use one of: {', '.join(SANDBOX_LEVELS)}."
        )
    if level == "unrestricted" and cfg["tools"].get("restrictToWorkspace"):
        level = "workspace"  # the setting from before 0.5 never widens access
    cfg["tools"]["sandboxLevel"] = level
    if cfg["agent"]["tier"] not in (None, *TIERS):
        raise ValueError(f"Invalid agent.tier {cfg['agent']['tier']!r}. Use one of: {', '.join(TIERS)}.")
    cfg["workspace"] = str(Path(cfg["workspace"] or agent_home() / "workspace").expanduser())
    return cfg


def read_config() -> dict | None:
    """The user's config file as written, or None when there is none."""
    path = config_file()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(config: dict) -> None:
    """Write the config file (0600: it can hold API keys)."""
    write_private(config_file(), json.dumps(config, indent=2))
