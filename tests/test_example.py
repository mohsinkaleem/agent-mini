"""examples/minimal_agent.py, the teaching example, run against a scripted fake model."""

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent / "examples" / "minimal_agent.py"


@pytest.fixture
def agent():
    spec = importlib.util.spec_from_file_location("minimal_agent", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(name: str, **args) -> dict:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(*replies: dict):
    """A fake ask_model that returns *replies* in order and records what it was sent."""
    sent = []

    def ask_model(messages):
        sent.append([dict(m) for m in messages])
        return replies[len(sent) - 1]
    return ask_model, sent


def test_runs_a_tool_and_returns_the_answer(agent, tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("buy milk")
    monkeypatch.chdir(tmp_path)
    ask_model, sent = scripted(
        {"role": "assistant", "content": None, "tool_calls": [call("read_file", path="notes.txt")]},
        {"role": "assistant", "content": "It says: buy milk."},
    )
    monkeypatch.setattr(agent, "ask_model", ask_model)

    assert agent.run_agent("What's in notes.txt?") == "It says: buy milk."
    tool_message = sent[1][-1]
    assert tool_message == {"role": "tool", "tool_call_id": "call_read_file", "content": "buy milk"}


def test_errors_go_back_to_the_model(agent, monkeypatch):
    ask_model, sent = scripted(
        {"role": "assistant", "content": None, "tool_calls": [call("python", code="1+1")]},
        {"role": "assistant", "content": None, "tool_calls": [call("read_file", path="missing.txt")]},
        {"role": "assistant", "content": "I couldn't find it."},
    )
    monkeypatch.setattr(agent, "ask_model", ask_model)

    assert agent.run_agent("Read missing.txt") == "I couldn't find it."
    assert sent[1][-1]["content"] == "Error: there is no tool called python."
    assert sent[2][-1]["content"].startswith("Error: ")


def test_shell_asks_before_running(agent, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert agent.shell("echo hi") == "The user declined to run this command."
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert agent.shell("echo hi") == "hi"


def test_stops_after_max_steps(agent, monkeypatch):
    looping = {"role": "assistant", "content": None, "tool_calls": [call("read_file", path="x")]}
    monkeypatch.setattr(agent, "ask_model", lambda messages: looping)
    assert agent.run_agent("loop").startswith(f"Stopped after {agent.MAX_STEPS} steps")
