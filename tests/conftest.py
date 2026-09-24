import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep the developer's ~/.agent-mini (plugins, memory) out of the tests."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENT_MINI_HOME", str(tmp_path / "home" / ".agent-mini"))
