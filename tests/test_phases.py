"""Tests for Phase 2–4 features: sessions, sandbox, fuzzy memory, plugins, vision, config."""

from pathlib import Path

import pytest

from agent_mini.agent.memory import Memory
from agent_mini.agent.tools import ToolExecutor
from agent_mini.sessions import (
    generate_session_id,
    list_sessions,
    load_session,
    save_session,
)

# ------------------------------------------------------------------
# Fuzzy memory matching (Phase 3.2)
# ------------------------------------------------------------------


def test_fuzzy_recall_synonym(tmp_path: Path):
    """Stemmed words should match — 'deployment' matches 'deploying'."""
    mem = Memory(tmp_path / "mem.json")
    mem.store("ops", "deploying the app to production")

    result = mem.recall("deployment")
    assert "deploying" in result


def test_fuzzy_recall_partial(tmp_path: Path):
    """Partial/substring matches should still score."""
    mem = Memory(tmp_path / "mem.json")
    mem.store("lang", "python programming")

    result = mem.recall("python")
    assert "python" in result


def test_fuzzy_recall_no_match(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("color", "blue sky")

    result = mem.recall("xyznonexistent")
    assert "No matching" in result


# ------------------------------------------------------------------
# Sandbox levels (Phase 3.3)
# ------------------------------------------------------------------


@pytest.fixture
def readonly_executor(tmp_path: Path) -> ToolExecutor:
    ws = tmp_path / "workspace"
    ws.mkdir()
    config = {
        "workspace": str(ws),
        "tools": {"sandboxLevel": "readonly"},
    }
    return ToolExecutor(config, Memory(tmp_path / "mem.json"))


@pytest.mark.asyncio
async def test_readonly_blocks_write(readonly_executor: ToolExecutor):
    result = await readonly_executor.execute("write_file", {"path": "x.txt", "content": "hi"})
    assert "blocked" in result.lower()
    assert "readonly" in result.lower()


@pytest.mark.asyncio
async def test_readonly_blocks_shell(readonly_executor: ToolExecutor):
    result = await readonly_executor.execute("shell_exec", {"command": "echo hi"})
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_readonly_blocks_code_edit(readonly_executor: ToolExecutor):
    result = await readonly_executor.execute(
        "code_edit", {"path": "x.txt", "old_text": "a", "new_text": "b"}
    )
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_readonly_allows_read(readonly_executor: ToolExecutor, tmp_path: Path):
    ws = tmp_path / "workspace"
    (ws / "test.txt").write_text("hello")
    result = await readonly_executor.execute("read_file", {"path": "test.txt"})
    assert result == "hello"


@pytest.mark.asyncio
async def test_readonly_allows_memory(readonly_executor: ToolExecutor):
    result = await readonly_executor.execute("memory_store", {"key": "k", "value": "v"})
    assert "Stored" in result


# ------------------------------------------------------------------
# Sessions (Phase 4.1)
# ------------------------------------------------------------------


def test_session_save_load():
    sid = generate_session_id()
    convo = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    save_session(sid, convo)

    loaded = load_session(sid)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]["content"] == "hello"


def test_session_list():
    save_session("s1", [{"role": "user", "content": "first"}])
    save_session("s2", [{"role": "user", "content": "second"}])

    sessions = list_sessions()
    assert len(sessions) == 2


def test_session_not_found():
    assert load_session("nonexistent") is None


# ------------------------------------------------------------------
# Vision utilities (Phase 4.4)
# ------------------------------------------------------------------


def test_vision_no_images():
    from agent_mini.agent.vision import build_image_content_parts
    result = build_image_content_parts("just some text without images")
    assert result is None


def test_vision_detects_url():
    from agent_mini.agent.vision import is_image_url
    assert is_image_url("https://example.com/photo.jpg") is True
    assert is_image_url("https://example.com/page.html") is False


def test_vision_detects_path():
    from agent_mini.agent.vision import is_image_path
    assert is_image_path("/tmp/photo.png") is True
    assert is_image_path("/tmp/data.json") is False


def test_vision_encode_image(tmp_path: Path):
    from agent_mini.agent.vision import encode_image_base64
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)  # minimal PNG header
    data, mime = encode_image_base64(img)
    assert mime == "image/png"
    assert len(data) > 0


def test_vision_build_parts_with_local_image(tmp_path: Path, monkeypatch):
    from agent_mini.agent.vision import build_image_content_parts
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    parts = build_image_content_parts(f"describe this {img}")
    assert parts is not None
    assert any(p.get("type") == "image_url" for p in parts)
    assert any(p.get("type") == "text" for p in parts)
