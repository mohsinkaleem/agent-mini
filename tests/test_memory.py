"""Tests for the Memory store."""

import json
from pathlib import Path

from agent_mini.agent.memory import Memory


def test_store_and_recall(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("project", "agent-mini is a personal AI agent")

    result = mem.recall("agent")
    assert "agent-mini" in result
    assert "project" in result


def test_recall_no_match(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("color", "blue")

    result = mem.recall("zzzzz_nonexistent")
    assert "No matching memories" in result


def test_get_recent(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("a", "first")
    mem.store("b", "second")
    mem.store("c", "third")

    recent = mem.get_recent(2)
    assert len(recent) == 2
    assert recent[-1]["key"] == "c"


def test_persistence(tmp_path: Path):
    filepath = tmp_path / "mem.json"
    mem1 = Memory(filepath)
    mem1.store("key1", "value1")

    # Load a fresh instance from the same file
    mem2 = Memory(filepath)
    result = mem2.recall("key1")
    assert "value1" in result


def test_max_entries(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json", max_entries=3)
    for i in range(5):
        mem.store(f"k{i}", f"v{i}")

    data = json.loads((tmp_path / "mem.json").read_text())
    assert len(data) == 3
    # Should keep the 3 most recent
    assert data[0]["key"] == "k2"


def test_corrupted_json_recovery(tmp_path: Path):
    """Corrupted memory file should be handled gracefully."""
    filepath = tmp_path / "mem.json"
    filepath.write_text("not valid json{{{")
    mem = Memory(filepath)
    assert mem._data == []
    # Should still work after recovery
    mem.store("key", "value")
    result = mem.recall("key")
    assert "value" in result


def test_recall_empty_query(tmp_path: Path):
    """Empty / whitespace query should return no matches."""
    mem = Memory(tmp_path / "mem.json")
    mem.store("key", "value")
    result = mem.recall("   ")
    assert "No matching" in result
