"""Tests for `agent-mini chat -m` flags and exit codes."""

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_mini import cli as cli_mod
from agent_mini.providers.base import BaseProvider, ChatResponse, ToolCall


class ScriptedProvider(BaseProvider):
    """Replays a fixed list of responses (or raises exceptions from it)."""

    def __init__(self, responses, model="qwen2.5:7b"):
        self._responses = list(responses)
        self._model = model

    async def chat(self, messages, tools=None, temperature=0.7):
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def name(self):
        return "scripted"

    @property
    def model_name(self):
        return self._model


@pytest.fixture
def run_chat(tmp_path: Path, monkeypatch):
    base_config = {
        "provider": "ollama",
        "providers": {"ollama": {"model": "qwen2.5:7b"}},
        "workspace": str(tmp_path / "ws"),
        "agent": {"maxIterations": 2},
    }

    def _run(provider, *args, config=None, message="hi", input=None):
        cfg = copy.deepcopy(config or base_config)
        seen: dict = {}

        def fake_create(c):
            seen["config"] = c
            return provider

        monkeypatch.setattr(cli_mod, "load_config", lambda: copy.deepcopy(cfg))
        monkeypatch.setattr(cli_mod, "MEMORY_FILE", tmp_path / "memory.json")
        monkeypatch.setattr("agent_mini.providers.create_provider", fake_create)
        result = CliRunner().invoke(
            cli_mod.cli, ["chat", "-m", message, "--no-markdown", *args], input=input
        )
        return result, seen

    return _run


def test_exit_ok_on_text_answer(run_chat):
    result, _ = run_chat(ScriptedProvider([ChatResponse(content="hello")]))
    assert result.exit_code == 0, result.output
    assert "hello" in result.output


def test_exit_2_on_max_iterations(run_chat):
    loop_call = ChatResponse(tool_calls=[ToolCall(id="1", name="list_directory", arguments={})])
    result, _ = run_chat(ScriptedProvider([loop_call]))
    assert result.exit_code == cli_mod.EXIT_MAX_ITERATIONS


def test_message_mode_prints_turn_stats(run_chat):
    responses = [
        ChatResponse(tool_calls=[ToolCall(id="1", name="list_directory", arguments={})]),
        ChatResponse(
            content="done",
            usage={"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
        ),
    ]
    result, _ = run_chat(ScriptedProvider(responses))
    assert result.exit_code == 0, result.output
    assert "2 iterations  •  1,200 in  •  30 out" in result.output


def test_exit_3_on_provider_error(run_chat):
    result, _ = run_chat(ScriptedProvider([ValueError("model 'x' not found")]))
    assert result.exit_code == cli_mod.EXIT_PROVIDER_ERROR


def test_provider_and_model_flags_override_config(run_chat):
    result, seen = run_chat(
        ScriptedProvider([ChatResponse(content="ok")]), "--provider", "openai", "--model", "gpt-5"
    )
    assert result.exit_code == 0, result.output
    assert seen["config"]["provider"] == "openai"
    assert seen["config"]["providers"]["openai"]["model"] == "gpt-5"


def test_invalid_sandbox_level_is_a_clean_config_error(run_chat, tmp_path):
    config = {
        "provider": "ollama",
        "workspace": str(tmp_path / "ws"),
        "tools": {"sandboxLevel": "read-only"},
    }
    result, _ = run_chat(ScriptedProvider([ChatResponse(content="ok")]), config=config)
    assert result.exit_code == 1
    assert "Config error" in result.output
    assert "Traceback" not in result.output


def test_exit_4_when_stuck(run_chat, tmp_path):
    config = {"provider": "ollama", "workspace": str(tmp_path / "ws"), "agent": {"maxIterations": 12}}
    loop_call = ChatResponse(tool_calls=[ToolCall(id="1", name="list_directory", arguments={})])
    result, _ = run_chat(ScriptedProvider([loop_call]), config=config)
    assert result.exit_code == cli_mod.EXIT_STUCK, result.output


def _write_then_answer():
    return ScriptedProvider([
        ChatResponse(tool_calls=[ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "x"})]),
        ChatResponse(content="done"),
    ])


def test_confirm_denies_when_nobody_can_answer(run_chat, tmp_path):
    ws = tmp_path / "ws"
    config = {"provider": "ollama", "workspace": str(ws), "tools": {"confirm": ["write_file"]}}
    result, _ = run_chat(_write_then_answer(), config=config)
    assert result.exit_code == 0, result.output
    assert "needs approval" in result.output
    assert not (ws / "a.txt").exists()


def test_confirm_prompt_accepts_yes(run_chat, tmp_path):
    ws = tmp_path / "ws"
    config = {"provider": "ollama", "workspace": str(ws), "tools": {"confirm": ["write_file"]}}
    result, _ = run_chat(_write_then_answer(), config=config, input="d\ny\n")
    assert result.exit_code == 0, result.output
    assert "+x" in result.output  # the diff was shown
    assert (ws / "a.txt").read_text() == "x"


def test_yes_flag_skips_approval(run_chat, tmp_path):
    ws = tmp_path / "ws"
    config = {"provider": "ollama", "workspace": str(ws), "tools": {"confirm": ["write_file"]}}
    result, seen = run_chat(_write_then_answer(), "--yes", config=config)
    assert result.exit_code == 0, result.output
    assert (ws / "a.txt").exists()
    assert seen["config"]["tools"]["confirm"] == []


def test_message_from_stdin(run_chat):
    provider = ScriptedProvider([ChatResponse(content="ok")])
    seen_messages = []
    original = provider.chat

    async def spy(messages, tools=None, temperature=0.7):
        seen_messages.append(messages[-1]["content"])
        return await original(messages, tools, temperature)

    provider.chat = spy
    result, _ = run_chat(provider, message="-", input="from a pipe\n")
    assert result.exit_code == 0, result.output
    assert seen_messages == ["from a pipe\n"]


def test_one_shot_runs_are_saved_as_sessions(run_chat):
    from agent_mini.sessions import list_sessions

    run_chat(ScriptedProvider([ChatResponse(content="hello")]))
    assert [s["preview"] for s in list_sessions()] == ["hi"]


# ── doctor (F6) ───────────────────────────────────────────────────────────────


def _doctor(tmp_path: Path, monkeypatch, config: dict | None):
    cfg_file = tmp_path / "config.json"
    if config is not None:
        cfg_file.write_text(json.dumps(config))
        cfg_file.chmod(0o644)
    monkeypatch.setattr(cli_mod, "CONFIG_FILE", cfg_file)
    monkeypatch.setattr(cli_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli_mod, "load_config", lambda: copy.deepcopy(config or {}))
    monkeypatch.setattr(cli_mod.console, "_width", 250)  # keep table rows on one line
    return CliRunner().invoke(cli_mod.cli, ["doctor"])


def test_doctor_without_config(tmp_path, monkeypatch):
    result = _doctor(tmp_path, monkeypatch, None)
    assert result.exit_code == 1
    assert "agent-mini init" in result.output


def test_doctor_flags_unreachable_provider_and_open_permissions(tmp_path, monkeypatch):
    config = {
        "provider": "ollama",
        "providers": {"ollama": {"baseUrl": "http://127.0.0.1:9", "model": "qwen3:8b"}},
        "workspace": str(tmp_path / "ws"),
    }
    result = _doctor(tmp_path, monkeypatch, config)
    assert result.exit_code == 1
    assert "unreachable" in result.output
    assert "chmod 600" in result.output
    assert "tier small" in result.output


def test_doctor_reports_config_errors(tmp_path, monkeypatch):
    config = {"provider": "ollama", "workspace": str(tmp_path / "ws"), "agent": {"tier": "huge"}}
    result = _doctor(tmp_path, monkeypatch, config)
    assert result.exit_code == 1
    assert "Invalid agent.tier" in result.output
