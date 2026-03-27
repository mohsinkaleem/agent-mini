"""Configuration loading and management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".agent-mini"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_WORKSPACE = CONFIG_DIR / "workspace"
MEMORY_FILE = CONFIG_DIR / "memory.json"


# ======================================================================
# Typed configuration dataclasses
# ======================================================================


@dataclass
class OllamaConfig:
    baseUrl: str = "http://localhost:11434"
    model: str = "llama3.1"
    think: bool | str | None = None


@dataclass
class GeminiConfig:
    apiKey: str = ""
    model: str = "gemini-2.0-flash"


@dataclass
class GitHubCopilotConfig:
    token: str = ""
    model: str = "gpt-4o"


@dataclass
class LocalConfig:
    baseUrl: str = "http://localhost:8080/v1"
    apiKey: str = "no-key"
    model: str = "local-model"


@dataclass
class ProvidersConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    github_copilot: GitHubCopilotConfig = field(default_factory=GitHubCopilotConfig)
    local: LocalConfig = field(default_factory=LocalConfig)


@dataclass
class AgentConfig:
    maxIterations: int = 20
    temperature: float = 0.7
    systemPrompt: str = ""


@dataclass
class TelegramConfig:
    enabled: bool = False
    token: str = ""
    allowFrom: list[str] = field(default_factory=list)
    streamResponses: bool = True


@dataclass
class ChannelsConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass
class ToolsConfig:
    restrictToWorkspace: bool = False
    sandboxLevel: str = "workspace"
    blockedCommands: list[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    enabled: bool = True
    maxEntries: int = 1000


@dataclass
class AppConfig:
    """Top-level typed configuration."""
    provider: str = "ollama"
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workspace: str = str(DEFAULT_WORKSPACE)

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        """Create an AppConfig from a raw dict, handling missing keys gracefully."""

        def _pick(raw: dict, dc: type) -> dict:
            return {k: v for k, v in raw.items() if k in dc.__dataclass_fields__}

        providers_raw = data.get("providers", {})
        return cls(
            provider=data.get("provider", "ollama"),
            providers=ProvidersConfig(
                ollama=OllamaConfig(**_pick(providers_raw.get("ollama", {}), OllamaConfig)),
                gemini=GeminiConfig(**_pick(providers_raw.get("gemini", {}), GeminiConfig)),
                github_copilot=GitHubCopilotConfig(
                    **_pick(providers_raw.get("github_copilot", {}), GitHubCopilotConfig)
                ),
                local=LocalConfig(**_pick(providers_raw.get("local", {}), LocalConfig)),
            ),
            agent=AgentConfig(**_pick(data.get("agent", {}), AgentConfig)),
            channels=ChannelsConfig(
                telegram=TelegramConfig(
                    **_pick(data.get("channels", {}).get("telegram", {}), TelegramConfig)
                ),
            ),
            tools=ToolsConfig(**_pick(data.get("tools", {}), ToolsConfig)),
            memory=MemoryConfig(**_pick(data.get("memory", {}), MemoryConfig)),
            workspace=data.get("workspace", str(DEFAULT_WORKSPACE)),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)


def load_config() -> dict:
    """Load config from ~/.agent-mini/config.json."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_typed_config() -> AppConfig | None:
    """Load config as a typed AppConfig. Returns None if no config exists."""
    raw = load_config()
    if not raw:
        return None
    return AppConfig.from_dict(raw)


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
