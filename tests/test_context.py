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


def test_build_system_prompt_includes_rules(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}

    prompt = build_system_prompt(config, mem)
    assert "<rules>" in prompt
    assert "read a file before editing" in prompt.lower()


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


# ── Tier-scaled prompt (P1) ─────────────────────────────────────────


def test_build_system_prompt_tier_tiny_is_compact(tmp_path: Path):
    """Tiny models get a shorter prompt with a compact rules block."""
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}

    tiny_prompt = build_system_prompt(config, mem, model_name="llama3.2:3b")
    small_prompt = build_system_prompt(config, mem, model_name="qwen2.5:7b")

    assert len(tiny_prompt) < len(small_prompt), "tiny prompt should be shorter"
    # Constraint-first anchor is still present
    assert "<rules>" in tiny_prompt


def test_build_system_prompt_tier_tiny_omits_memory(tmp_path: Path):
    """Tiny tier gets zero memory entries — noise budget matters."""
    mem = Memory(tmp_path / "mem.json")
    mem.store("k1", "v1")
    mem.store("k2", "v2")
    config = {"workspace": str(tmp_path)}

    tiny_prompt = build_system_prompt(config, mem, model_name="llama3.2:3b")
    cloud_prompt = build_system_prompt(config, mem, model_name="gpt-4o")

    assert "Recent memories" not in tiny_prompt
    assert "Recent memories" in cloud_prompt


# ── Inline tool list (P2) ───────────────────────────────────────────


def test_build_system_prompt_includes_tool_list(tmp_path: Path):
    """When tool_defs is provided, an <available_tools> block is inlined."""
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}
    tool_defs = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a file."}},
        {"type": "function", "function": {"name": "shell_exec", "description": "Run a shell command."}},
    ]
    prompt = build_system_prompt(config, mem, tool_defs=tool_defs)
    assert "<available_tools>" in prompt
    assert "read_file" in prompt
    assert "shell_exec" in prompt


def test_build_system_prompt_no_tool_list_when_empty(tmp_path: Path):
    mem = Memory(tmp_path / "mem.json")
    config = {"workspace": str(tmp_path)}
    prompt = build_system_prompt(config, mem, tool_defs=[])
    assert "<available_tools>" not in prompt
