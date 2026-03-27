"""Tests for config loading."""

from pathlib import Path

from agent_mini.config import get_workspace, load_config


def test_load_config_missing_file(monkeypatch):
    """load_config returns empty dict when config file doesn't exist."""
    import agent_mini.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_FILE", Path("/tmp/nonexistent_agent_mini_cfg.json"))
    assert load_config() == {}


def test_get_workspace_default():
    """get_workspace returns a Path and creates the directory."""
    ws = get_workspace({})
    assert isinstance(ws, Path)
    assert ws.exists()


def test_get_workspace_custom(tmp_path: Path):
    ws_dir = tmp_path / "my_workspace"
    config = {"workspace": str(ws_dir)}
    ws = get_workspace(config)
    assert ws == ws_dir
    assert ws.exists()
