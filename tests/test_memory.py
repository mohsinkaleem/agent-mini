"""The memory store."""

import json
from pathlib import Path

from agent_mini.memory import Memory


def test_store_and_recall(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("project", "agent-mini is a personal AI agent")
    mem.store("color", "blue sky")
    assert mem.recall("agent") == "project: agent-mini is a personal AI agent"
    assert mem.recall("xyznonexistent") == "No matching memories found."
    assert mem.recall("   ") == "No matching memories found."


def test_recall_matches_word_forms_and_ranks(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("ops", "deploying the app to production")
    mem.store("users", "three users share the laptop")
    mem.store("editor", "favourite text editor is helix")
    assert "deploying" in mem.recall("deployment")
    assert "three users" in mem.recall("user")
    assert mem.recall("which text editor").splitlines()[0] == "editor: favourite text editor is helix"


def test_persistence_and_limit(tmp_path: Path):
    path = tmp_path / "mem.json"
    mem = Memory(path, max_entries=3)
    for i in range(5):
        mem.store(f"k{i}", f"v{i}")
    assert [e["key"] for e in json.loads(path.read_text())] == ["k2", "k3", "k4"]
    assert [e["key"] for e in Memory(path).recent(2)] == ["k3", "k4"]
    assert Memory(path).recent(0) == []
    assert path.stat().st_mode & 0o777 == 0o600


def test_corrupted_file_is_kept_aside(tmp_path: Path):
    path = tmp_path / "mem.json"
    path.write_text("not valid json{{{")
    mem = Memory(path)
    assert mem.recent() == []
    assert (tmp_path / "mem.json.corrupt").read_text() == "not valid json{{{"
    mem.store("key", "value")
    assert mem.recall("key") == "key: value"


def test_same_key_updates_and_forget(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    assert mem.store("Editor", "vim") == "Stored memory: Editor"
    assert mem.store("editor ", "helix") == "Updated memory: editor "
    assert [(e["key"], e["value"]) for e in mem.recent(10)] == [("editor ", "helix")]
    mem.store("b", "2")
    assert mem.forget("EDITOR") == "Forgot memory: EDITOR"
    assert mem.forget("missing").startswith("No memory")
    assert [e["key"] for e in Memory(tmp_path / "mem.json").recent(10)] == ["b"]
