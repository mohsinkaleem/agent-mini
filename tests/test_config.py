"""Tests for config loading."""

import json
from pathlib import Path

from agent_mini.config import agent_home, load_config, save_config


def test_load_config_missing_file(monkeypatch):
    """load_config returns empty dict when config file doesn't exist."""
    import agent_mini.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_FILE", Path("/tmp/nonexistent_agent_mini_cfg.json"))
    assert load_config() == {}


def test_save_config_is_private(tmp_path: Path, monkeypatch):
    """S5: the config holds API keys, so only the owner may read it."""
    import agent_mini.config as cfg

    path = tmp_path / "cfg" / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_FILE", path)
    save_config({"providers": {"openai": {"apiKey": "sk-x"}}})
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["providers"]["openai"]["apiKey"] == "sk-x"


def test_agent_home_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MINI_HOME", str(tmp_path / "alt"))
    assert agent_home() == tmp_path / "alt"
    monkeypatch.delenv("AGENT_MINI_HOME")
    assert agent_home() == Path.home() / ".agent-mini"


def test_api_keys_fall_back_to_env(monkeypatch):
    from agent_mini.providers import create_provider

    monkeypatch.delenv("AGENT_MINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    provider = create_provider({"provider": "openai"})
    assert provider._headers["Authorization"] == "Bearer sk-env"
    monkeypatch.setenv("AGENT_MINI_API_KEY", "am-env")
    assert create_provider({"provider": "local"})._headers["Authorization"] == "Bearer am-env"
    configured = create_provider({"provider": "openai", "providers": {"openai": {"apiKey": "sk-cfg"}}})
    assert configured._headers["Authorization"] == "Bearer sk-cfg"
