from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_mini.memory import Memory
from agent_mini.tools import Tools


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep the developer's ~/.agent-mini (config, plugins, memory) out of the tests."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENT_MINI_HOME", str(tmp_path / "home" / ".agent-mini"))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "mem.json")


@pytest.fixture
def config(workspace: Path) -> dict:
    return {"workspace": str(workspace), "agent": {"maxIterations": 5}}


@pytest.fixture
async def tools(workspace: Path, memory: Memory):
    t = Tools({"workspace": str(workspace)}, memory)
    yield t
    await t.close()


def fake_provider(responses, model: str = "qwen2.5:7b") -> AsyncMock:
    """A provider whose chat() returns (or raises) *responses* in order."""
    provider = AsyncMock()
    provider.chat = AsyncMock(side_effect=responses)
    provider.model = model
    provider.model_info = AsyncMock(return_value=None)
    return provider
