"""`agent-mini chat -m` flags, exit codes and reports, and `doctor`."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_mini import cli as cli_mod
from agent_mini.config import config_file, read_config, save_config
from agent_mini.providers.base import ChatResponse, Provider, ToolCall


class ScriptedProvider(Provider):
    """Replays a list of responses (raising the exceptions in it); the last one repeats."""

    name = "scripted"

    def __init__(self, responses, model="qwen2.5:7b"):
        super().__init__(model)
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    async def chat(self, messages, tools=None, temperature=None, on_delta=None, on_thinking=None):
        self.requests.append(messages)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def run_chat(tmp_path: Path, monkeypatch):
    def run(provider, *args, config=None, message="hi", input=None):
        save_config(config or {"workspace": str(tmp_path / "ws"), "agent": {"maxIterations": 2}})
        seen: dict = {}

        def fake_create(cfg):
            seen["config"] = cfg
            return provider

        monkeypatch.setattr("agent_mini.providers.create_provider", fake_create)
        result = CliRunner().invoke(cli_mod.cli, ["chat", "-m", message, "--no-markdown", *args], input=input)
        return result, seen

    return run


def _list_forever():
    return ScriptedProvider([ChatResponse(tool_calls=[ToolCall(id="1", name="list_files", arguments={})])])


def test_text_answer_exits_0(run_chat):
    result, _ = run_chat(ScriptedProvider([ChatResponse(content="hello")]))
    assert result.exit_code == 0, result.output
    assert "hello" in result.output


def test_exit_codes(run_chat, tmp_path):
    assert run_chat(_list_forever())[0].exit_code == cli_mod.EXIT_MAX_ITERATIONS
    assert run_chat(ScriptedProvider([ValueError("model 'x' not found")]))[0].exit_code == cli_mod.EXIT_PROVIDER_ERROR
    stuck = {"workspace": str(tmp_path / "ws"), "agent": {"maxIterations": 12}}
    assert run_chat(_list_forever(), config=stuck)[0].exit_code == cli_mod.EXIT_STUCK


def test_turn_stats(run_chat):
    responses = [ChatResponse(tool_calls=[ToolCall(id="1", name="list_files", arguments={})]),
                 ChatResponse(content="done", usage={"prompt_tokens": 1200, "completion_tokens": 30,
                                                     "total_tokens": 1230})]
    result, _ = run_chat(ScriptedProvider(responses))
    assert "2 iterations  •  1,200 in  •  30 out" in result.output


def test_report(run_chat, tmp_path):
    report = tmp_path / "report.json"
    responses = [ChatResponse(tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "x"})]),
                 ChatResponse(content="done")]
    result, _ = run_chat(ScriptedProvider(responses), "--report", str(report))
    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    assert data["reply"] == "done" and data["finish_reason"] == "text" and data["tier"] == "small"
    assert data["tool_calls"] == [{"name": "read_file", "args": {"path": "x"}, "error": True}]
    assert data["session"] and data["messages"][0]["role"] == "system"


def test_flags_override_the_config_for_one_run(run_chat, tmp_path):
    result, seen = run_chat(ScriptedProvider([ChatResponse(content="ok")]), "--provider", "openai",
                            "--model", "gpt-5", "--workspace", str(tmp_path / "other"))
    assert result.exit_code == 0, result.output
    assert seen["config"]["provider"] == "openai"
    assert seen["config"]["providers"]["openai"]["model"] == "gpt-5"
    assert seen["config"]["workspace"] == str(tmp_path / "other")
    assert "openai" not in json.dumps(read_config())  # nothing was saved


def test_config_errors_exit_cleanly(run_chat, tmp_path):
    result, _ = run_chat(ScriptedProvider([ChatResponse(content="ok")]),
                         config={"workspace": str(tmp_path / "ws"), "tools": {"sandboxLevel": "read-only"}})
    assert result.exit_code == 1
    assert "Config error" in result.output and "Traceback" not in result.output


def test_missing_config(tmp_path):
    result = CliRunner().invoke(cli_mod.cli, ["chat", "-m", "hi"])
    assert result.exit_code == 1 and "agent-mini init" in result.output


def _write_then_answer():
    return ScriptedProvider([
        ChatResponse(tool_calls=[ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "x"})]),
        ChatResponse(content="done"),
    ])


def _confirming(tmp_path):
    return {"workspace": str(tmp_path / "ws"), "tools": {"confirm": ["write_file"]}}


def test_confirm_denies_when_nobody_can_answer(run_chat, tmp_path):
    result, _ = run_chat(_write_then_answer(), config=_confirming(tmp_path))
    assert result.exit_code == 0, result.output
    assert "needs approval" in result.output
    assert not (tmp_path / "ws" / "a.txt").exists()


def test_confirm_prompt_shows_a_diff_and_accepts_yes(run_chat, tmp_path):
    result, _ = run_chat(_write_then_answer(), config=_confirming(tmp_path), input="d\ny\n")
    assert result.exit_code == 0, result.output
    assert "+x" in result.output
    assert (tmp_path / "ws" / "a.txt").read_text() == "x"


def test_yes_flag_skips_approval(run_chat, tmp_path):
    result, seen = run_chat(_write_then_answer(), "--yes", config=_confirming(tmp_path))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ws" / "a.txt").exists()
    assert seen["config"]["tools"]["confirm"] == []


def test_message_from_stdin(run_chat):
    provider = ScriptedProvider([ChatResponse(content="ok")])
    result, _ = run_chat(provider, message="-", input="from a pipe\n")
    assert result.exit_code == 0, result.output
    assert provider.requests[0][-1]["content"] == "from a pipe\n"


def test_one_shot_runs_are_saved_as_sessions(run_chat):
    from agent_mini.sessions import list_sessions

    run_chat(ScriptedProvider([ChatResponse(content="hello")]))
    assert [s["preview"] for s in list_sessions()] == ["hi"]


def test_init_writes_only_the_choices(tmp_path):
    answers = "\n".join(["1", "", "qwen3.5:4b", str(tmp_path / "ws"), "y"]) + "\n"
    result = CliRunner().invoke(cli_mod.cli, ["init"], input=answers)
    assert result.exit_code == 0, result.output
    assert read_config() == {
        "provider": "ollama",
        "providers": {"ollama": {"baseUrl": "http://localhost:11434", "model": "qwen3.5:4b"}},
        "workspace": str(tmp_path / "ws"),
        "tools": {"confirm": ["shell", "write_file", "edit_file"]},
    }
    assert (tmp_path / "ws").is_dir()


# ── doctor ───────────────────────────────────────────────────────────


def _doctor(monkeypatch, config: dict | None):
    if config is not None:
        save_config(config)
        config_file().chmod(0o644)
    monkeypatch.setattr(cli_mod.console, "_width", 250)  # keep table rows on one line
    return CliRunner().invoke(cli_mod.cli, ["doctor"])


def test_doctor_without_config(monkeypatch):
    result = _doctor(monkeypatch, None)
    assert result.exit_code == 1 and "agent-mini init" in result.output


def test_doctor_flags_unreachable_provider_and_open_permissions(tmp_path, monkeypatch):
    config = {"providers": {"ollama": {"baseUrl": "http://127.0.0.1:9", "model": "qwen3:8b"}},
              "workspace": str(tmp_path / "ws")}
    result = _doctor(monkeypatch, config)
    assert result.exit_code == 1
    assert "unreachable" in result.output and "chmod 600" in result.output and "tier small" in result.output


def test_doctor_reports_config_errors(tmp_path, monkeypatch):
    result = _doctor(monkeypatch, {"workspace": str(tmp_path / "ws"), "agent": {"tier": "huge"}})
    assert result.exit_code == 1 and "Invalid agent.tier" in result.output


def test_piped_output_holds_only_the_reply(run_chat):
    responses = [ChatResponse(tool_calls=[ToolCall(id="1", name="list_files", arguments={})]),
                 ChatResponse(content="Answer [exit code: 1] stays.")]
    result, _ = run_chat(ScriptedProvider(responses))
    assert result.stdout.strip() == "Answer [exit code: 1] stays."  # no header, tool lines, or eaten [markup]
    assert "Agent Mini" in result.stderr and "list_files" in result.stderr


def test_a_per_run_workspace_is_a_sandbox(run_chat, tmp_path):
    config = {"workspace": str(tmp_path / "ws"), "tools": {"sandboxLevel": "unrestricted"}}
    _, seen = run_chat(ScriptedProvider([ChatResponse(content="ok")]), "--workspace", str(tmp_path / "w2"),
                       config=config)
    assert seen["config"]["tools"]["sandboxLevel"] == "workspace"
