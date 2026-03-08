"""Tests for ToolExecutor — file ops, code_edit, shell blocklist, path resolution."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from agent_mini.agent.memory import Memory
from agent_mini.agent.tools import ToolExecutor


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "mem.json")


@pytest.fixture
def executor(workspace: Path, memory: Memory) -> ToolExecutor:
    config = {
        "workspace": str(workspace),
        "tools": {"restrictToWorkspace": True},
    }
    return ToolExecutor(config, memory)


@pytest.fixture
def unrestricted_executor(workspace: Path, memory: Memory) -> ToolExecutor:
    config = {
        "workspace": str(workspace),
        "tools": {"restrictToWorkspace": False},
    }
    return ToolExecutor(config, memory)


# ------------------------------------------------------------------
# File operations
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_and_read_file(executor: ToolExecutor, workspace: Path):
    result = await executor.execute("write_file", {"path": "test.txt", "content": "hello world"})
    assert "Written" in result

    result = await executor.execute("read_file", {"path": "test.txt"})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_append_file(executor: ToolExecutor, workspace: Path):
    await executor.execute("write_file", {"path": "log.txt", "content": "line1\n"})
    await executor.execute("append_file", {"path": "log.txt", "content": "line2\n"})

    result = await executor.execute("read_file", {"path": "log.txt"})
    assert "line1" in result
    assert "line2" in result


@pytest.mark.asyncio
async def test_list_directory(executor: ToolExecutor, workspace: Path):
    (workspace / "a.txt").write_text("a")
    (workspace / "subdir").mkdir()

    result = await executor.execute("list_directory", {"path": "."})
    assert "subdir" in result
    assert "a.txt" in result


# ------------------------------------------------------------------
# code_edit
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_edit_success(executor: ToolExecutor, workspace: Path):
    (workspace / "hello.py").write_text("print('hello')\nprint('world')\n")

    result = await executor.execute(
        "code_edit",
        {"path": "hello.py", "old_text": "print('hello')", "new_text": "print('hi')"},
    )
    assert "Edited" in result
    assert "replaced 1 occurrence" in result

    content = (workspace / "hello.py").read_text()
    assert "print('hi')" in content
    assert "print('world')" in content


@pytest.mark.asyncio
async def test_code_edit_not_found(executor: ToolExecutor, workspace: Path):
    (workspace / "f.py").write_text("abc")

    result = await executor.execute(
        "code_edit",
        {"path": "f.py", "old_text": "xyz", "new_text": "123"},
    )
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_code_edit_multiple_matches(executor: ToolExecutor, workspace: Path):
    (workspace / "dup.py").write_text("aaa\naaa\n")

    result = await executor.execute(
        "code_edit",
        {"path": "dup.py", "old_text": "aaa", "new_text": "bbb"},
    )
    assert "Error" in result
    assert "2 locations" in result


@pytest.mark.asyncio
async def test_code_edit_file_not_found(executor: ToolExecutor, workspace: Path):
    result = await executor.execute(
        "code_edit",
        {"path": "nonexistent.py", "old_text": "a", "new_text": "b"},
    )
    assert "Error" in result
    assert "not found" in result


# ------------------------------------------------------------------
# Path restriction
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restricted_path_blocks_outside_workspace(executor: ToolExecutor):
    result = await executor.execute("read_file", {"path": "/etc/passwd"})
    assert "Error" in result
    assert "outside workspace" in result.lower() or "Access denied" in result


# ------------------------------------------------------------------
# Shell blocklist
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shell_blocks_rm_rf(executor: ToolExecutor):
    result = await executor.execute("shell_exec", {"command": "rm -rf /"})
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_shell_blocks_sudo(executor: ToolExecutor):
    result = await executor.execute("shell_exec", {"command": "sudo apt install foo"})
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_shell_blocks_fork_bomb(executor: ToolExecutor):
    result = await executor.execute("shell_exec", {"command": ":() { :|:& };:"})
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_shell_allows_safe_commands(executor: ToolExecutor, workspace: Path):
    (workspace / "test.txt").write_text("hello")
    result = await executor.execute("shell_exec", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_shell_custom_blocklist(workspace: Path, memory: Memory):
    config = {
        "workspace": str(workspace),
        "tools": {
            "restrictToWorkspace": False,
            "blockedCommands": [r"\bcurl\b"],
        },
    }
    ex = ToolExecutor(config, memory)
    result = await ex.execute("shell_exec", {"command": "curl http://example.com"})
    assert "blocked" in result.lower()


# ------------------------------------------------------------------
# Memory tools
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_and_recall(executor: ToolExecutor):
    result = await executor.execute("memory_store", {"key": "lang", "value": "python"})
    assert "Stored" in result

    result = await executor.execute("memory_recall", {"query": "lang"})
    assert "python" in result


# ------------------------------------------------------------------
# Unknown tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool(executor: ToolExecutor):
    result = await executor.execute("nonexistent_tool", {})
    assert "Unknown tool" in result


@pytest.mark.asyncio
async def test_path_traversal_blocked(executor: ToolExecutor, workspace: Path):
    """Relative path traversal should be blocked when restricted."""
    result = await executor.execute("read_file", {"path": "../../etc/passwd"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_list_directory_uses_text_prefixes(executor: ToolExecutor, workspace: Path):
    """list_directory should use text prefixes, not emoji."""
    (workspace / "subdir").mkdir()
    (workspace / "file.txt").write_text("x")
    result = await executor.execute("list_directory", {"path": "."})
    assert "[dir]" in result
    assert "[file]" in result


@pytest.mark.asyncio
async def test_web_search_returns_results(executor: ToolExecutor):
    """web_search should handle empty results gracefully."""
    # We can't test actual DDG, but we test the tool doesn't crash
    import httpx
    from unittest.mock import AsyncMock, patch

    mock_response = AsyncMock()
    mock_response.text = "<html><body>No results</body></html>"
    mock_response.raise_for_status = lambda: None

    with patch.object(executor._http, "post", return_value=mock_response):
        result = await executor.execute("web_search", {"query": "test query"})
    assert isinstance(result, str)
