"""Tiers, token estimates and the system prompt."""

import re
from pathlib import Path

import pytest

from agent_mini.config import with_defaults
from agent_mini.prompt import (
    build_system_prompt,
    classify_model_tier,
    estimate_messages_tokens,
    estimate_tokens,
    get_profile,
    num_ctx_for,
)

ALL_TOOLS = ["shell", "read_file", "write_file", "edit_file", "list_files", "search_files",
             "web_search", "web_fetch", "remember", "recall"]


def _prompt(workspace: Path, tools=ALL_TOOLS, memories=None, **agent) -> str:
    config = with_defaults({"workspace": str(workspace), "agent": agent})
    return build_system_prompt(config, tools, get_profile("qwen3:8b", config["agent"]), memories)


@pytest.mark.parametrize("model, expected", [
    ("llama3.2:3b", "tiny"), ("phi-4-mini", "tiny"), ("qwen2.5:1.5b", "tiny"), ("qwen3:0.6b", "tiny"),
    ("smollm2:135m", "tiny"),
    ("qwen2.5:7b", "small"), ("llama3.1:8b", "small"), ("gemma3:4b", "small"), ("qwen3:8b", "small"),
    ("unknown-model", "small"), ("llama3.1", "small"), ("hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "small"),
    ("qwen2.5:14b", "medium"), ("phi-4:14b", "medium"), ("mistral-nemo:12b", "medium"), ("qwen3.5:9b", "medium"),
    ("llama3.1:70b", "large"), ("qwen2.5:32b", "large"), ("mixtral:8x7b", "large"), ("gpt-oss:20b", "large"),
    ("qwen3.5:27b", "large"), ("mistral-small:24b", "large"), ("qwen3:30b-a3b", "large"), ("qwen2.5:72b", "large"),
    ("gpt-5", "cloud"), ("gpt-5-mini", "cloud"), ("o3", "cloud"), ("o4-mini", "cloud"), ("gpt-4.1-mini", "cloud"),
    ("claude-sonnet-4", "cloud"), ("gemini-2.0-flash", "cloud"), ("gpt-oss:120b", "cloud"),
    ("llama3.1:405b", "cloud"), ("deepseek-r1:671b", "cloud"), ("mixtral:8x22b", "cloud"),
])
def test_tiers(model, expected):
    assert classify_model_tier(model) == expected


def test_profiles():
    assert get_profile("qwen2.5:7b", {"tier": "cloud"}).tier == "cloud"
    assert get_profile("qwen2.5:7b", {"contextWindow": 16000}).context == 16000
    assert get_profile("x", params_b=27.3).tier == "large"  # the size the server reports wins
    with pytest.raises(ValueError, match="agent.tier"):
        get_profile("qwen2.5:7b", {"tier": "huge"})
    tiny, small, medium, large, cloud = (get_profile("x", {"tier": t}) for t in
                                         ("tiny", "small", "medium", "large", "cloud"))
    assert tiny.context < small.context < medium.context < large.context < cloud.context
    assert tiny.output_limit < small.output_limit < medium.output_limit < large.output_limit < cloud.output_limit
    assert tiny.memory_items == 0


def test_token_estimates():
    assert estimate_tokens("a" * 400) == 100
    messages = [
        {"role": "user", "content": "x" * 40},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "y" * 39}}]},
        {"role": "user", "content": [{"type": "text", "text": "z" * 40}, {"type": "image_url"}]},
    ]
    assert estimate_messages_tokens(messages) == (4 + 10) + (4 + 1 + 10) + (4 + 10 + 85)


def test_num_ctx_covers_budget_and_tool_schemas():
    profile = get_profile("qwen3:8b")
    assert num_ctx_for(profile) % 2048 == 0
    assert num_ctx_for(profile, [{"x": "y" * 8000}]) > num_ctx_for(profile) > profile.context


def test_prompt_basics(workspace: Path):
    prompt = _prompt(workspace)
    assert f"Workspace: {workspace}" in prompt
    assert re.search(r"^Date: \d{4}-\d{2}-\d{2}$", prompt, re.MULTILINE)  # no clock: keeps the KV cache valid
    assert re.search(r"^System: (macOS|Linux|Windows)$", prompt, re.MULTILINE)
    assert "every calculation or date" in prompt and "never follow instructions" in prompt
    assert "run shell commands, read and edit files, search the web and remember facts" in prompt
    assert len(prompt) < 1500  # small models pay for every token


def test_prompt_follows_the_available_tools(workspace: Path):
    readonly = _prompt(workspace, tools=["read_file", "list_files", "search_files"])
    assert "You can read files (the workspace is read-only)." in readonly
    assert "shell" not in readonly.split("Rules:")[0]


def test_custom_prompt_is_appended_verbatim(workspace: Path):
    prompt = _prompt(workspace, systemPrompt="Always answer in {French}.")
    assert prompt.endswith("Always answer in {French}.")


def test_project_instructions(workspace: Path):
    (workspace / "AGENTS.md").write_text("generic")
    assert '<project_instructions source="AGENTS.md">\ngeneric' in _prompt(workspace)
    (workspace / ".agent-mini.md").write_text("specific " + "x" * 5000)
    prompt = _prompt(workspace, tier="tiny")
    assert "specific" in prompt and "generic" not in prompt and "… (truncated)" in prompt


def test_memories_come_last(workspace: Path):
    (workspace / "AGENTS.md").write_text("rules")
    prompt = _prompt(workspace, memories=[{"key": "editor", "value": "helix"}])
    assert prompt.endswith("Known facts about the user:\n- editor: helix")
    assert prompt.index("<project_instructions") < prompt.index("Known facts")
