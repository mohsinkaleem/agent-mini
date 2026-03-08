"""Tests for the system prompt builder."""

from pathlib import Path

from agent_mini.agent.context import build_system_prompt
from agent_mini.agent.memory import Memory


def test_build_system_prompt_includes_workspace(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}

    prompt = build_system_prompt(config, mem)
    assert str(tmp_path) in prompt
    assert "Agent Mini" in prompt


def test_build_system_prompt_includes_code_edit(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}

    prompt = build_system_prompt(config, mem)
    assert "code_edit" in prompt


def test_build_system_prompt_includes_custom_prompt(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    config = {
        "workspace": str(tmp_path),
        "agent": {"systemPrompt": "Always respond in French."},
    }

    prompt = build_system_prompt(config, mem)
    assert "Always respond in French" in prompt


def test_build_system_prompt_includes_recent_memories(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    mem.store("project", "agent-mini")

    config = {"workspace": str(tmp_path)}
    prompt = build_system_prompt(config, mem)
    assert "agent-mini" in prompt
    assert "Recent memories" in prompt


def test_build_system_prompt_format_string_injection(tmp_path: Path):
    """User systemPrompt with {braces} must not cause format string errors."""
    mem = Memory(tmp_path / "mem.json")
    config = {
        "workspace": str(tmp_path),
        "agent": {"systemPrompt": "Use {model} and {date} format"},
    }
    # Should not raise KeyError or inject variables
    prompt = build_system_prompt(config, mem)
    assert "{model}" in prompt  # Literal braces preserved
    assert "{date}" in prompt


def test_build_system_prompt_no_memories(tmp_path: Path):
    """No memories should not produce empty 'Recent memories' section."""
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}
    prompt = build_system_prompt(config, mem)
    assert "Recent memories" not in prompt
