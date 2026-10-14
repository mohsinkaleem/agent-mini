"""Config defaults, validation and files."""

import json
from pathlib import Path

import pytest

from agent_mini.config import DEFAULTS, agent_home, config_file, read_config, save_config, with_defaults

ROOT = Path(__file__).parent.parent


def test_defaults_fill_in_and_user_values_win():
    cfg = with_defaults({"providers": {"ollama": {"model": "qwen3.5:4b"}}, "tools": {"sandboxLevel": "ReadOnly"}})
    assert cfg["providers"]["ollama"] == {**DEFAULTS["providers"]["ollama"], "model": "qwen3.5:4b"}
    assert cfg["tools"]["sandboxLevel"] == "readonly"
    assert cfg["tools"]["shellTimeout"] == 120
    assert cfg["workspace"] == str(agent_home() / "workspace")
    assert with_defaults(cfg) == cfg  # idempotent
    legacy = {"tools": {"sandboxLevel": "unrestricted", "restrictToWorkspace": True}}
    assert with_defaults(legacy)["tools"]["sandboxLevel"] == "workspace"  # never widens access


@pytest.mark.parametrize("bad, message", [
    ({"provider": "anthropic"}, "Unknown provider"),
    ({"tools": {"sandboxLevel": "read-only"}}, "sandboxLevel"),
    ({"agent": {"tier": "huge"}}, "agent.tier"),
])
def test_invalid_values_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        with_defaults(bad)


def test_example_config_matches_the_defaults():
    example = json.loads((ROOT / "config.example.json").read_text())
    assert example == {**DEFAULTS, "workspace": "~/.agent-mini/workspace"}


def test_config_file_round_trip_is_private():
    assert read_config() is None
    save_config({"providers": {"openai": {"apiKey": "sk-x"}}})
    assert config_file().stat().st_mode & 0o777 == 0o600
    assert read_config()["providers"]["openai"]["apiKey"] == "sk-x"


def test_agent_home_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MINI_HOME", str(tmp_path / "alt"))
    assert agent_home() == tmp_path / "alt"
    monkeypatch.delenv("AGENT_MINI_HOME")
    assert agent_home() == Path.home() / ".agent-mini"
