"""The eval harness, without an LLM."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_RUN = Path(__file__).parent.parent / "evals" / "run.py"
_spec = importlib.util.spec_from_file_location("eval_run", _RUN)
run = importlib.util.module_from_spec(_spec)
sys.modules["eval_run"] = run
_spec.loader.exec_module(run)


def test_every_task_parses_and_its_checks_are_valid():
    """Each task's checks fail on the untouched fixture and pass on its reference solution."""
    tasks = run.load_tasks()
    assert len(tasks) == len(list(run.TASKS_DIR.glob("*.toml")))
    assert run.validate(tasks) == []


def test_unknown_fields_are_rejected(tmp_path: Path):
    path = tmp_path / "t.toml"
    path.write_text('id = "t"\nprompt = "p"\nexpect = "x"\n')
    with pytest.raises(ValueError, match="unknown field"):
        run.parse_task(path)


def test_matchers(tmp_path: Path):
    exact = run.Check(shell="echo 10", expect_exact="0")
    assert "expected exactly" in run.run_check(exact, tmp_path, "")
    answer = run.Check(expect_regex=r"\b391\b")
    assert run.run_check(answer, tmp_path, "It is 391.") == ""
    assert run.run_check(answer, tmp_path, "3910") != ""
    assert run.run_check(run.Check(expect_no_stdout="Priya"), tmp_path, "Marco, not Priya") != ""
    python = run.Check(shell="{python} -c 'import sys; print(sys.executable)'")
    assert run.run_check(python, tmp_path, "") == ""


def test_judge_limits(tmp_path: Path):
    task = run.Task(id="t", prompt="p", max_tool_calls=1, forbidden_tools=["web_search"])
    calls = [{"name": "read_file"}, {"name": "read_file"}]
    assert run.judge(task, tmp_path, "", calls) == "too many tool calls: 2 > 1"
    assert run.judge(task, tmp_path, "", [{"name": "web_search"}]) == "forbidden tool: web_search"
    assert run.judge(task, tmp_path, "", [{"name": "read_file"}]) == ""


@pytest.mark.parametrize("tamper", [False, True])
def test_fixture_unchanged(tamper: bool):
    task = next(t for t in run.load_tasks() if t.id == "fix_failing_test")
    ws = run.prepare_workspace(task)
    (ws / "tests" / "__pycache__").mkdir(exist_ok=True)  # caches don't count
    if tamper:
        (ws / "tests" / "test_stack.py").write_text("def test_ok():\n    pass\n")
    assert run.tampered(task, ws) == ("tests" if tamper else "")


def test_eval_home_holds_provider_settings_task_config_and_overrides(tmp_path: Path, monkeypatch):
    user_home = tmp_path / "user"
    user_home.mkdir()
    (user_home / "config.json").write_text(json.dumps({
        "provider": "ollama",
        "providers": {"ollama": {"model": "qwen3.5:4b"}},
        "agent": {"systemPrompt": "be a pirate", "temperature": 1.5},
        "memory": {"enabled": True},
    }))
    monkeypatch.setenv("AGENT_MINI_HOME", str(user_home))
    task = run.Task(id="t", prompt="p", config={"memory": {"enabled": True}})
    home = run.prepare_home(tmp_path / "ws", task, run._dotted(["agent.temperature=0.2", "agent.tier=tiny"]))
    config = json.loads((home / "config.json").read_text())
    assert config["providers"] == {"ollama": {"model": "qwen3.5:4b"}}
    assert config["agent"] == {"temperature": 0.2, "tier": "tiny"}  # no system prompt of the user's
    assert config["memory"] == {"enabled": True}
    assert config["workspace"] == str(tmp_path / "ws")


def test_run_task_scores_a_fake_agent(monkeypatch):
    task = next(t for t in run.load_tasks() if t.id == "followup_edit")
    turns = []

    def fake_agent(prompt, ws, home, args, timeout):
        turns.append(args)
        source = "def add(a, b):\n    return a + b\n"
        if len(turns) == 2:
            source += "\n\ndef sub(a, b):\n    return a - b\n"
        (ws / "calc.py").write_text(source)
        report = {"session": "s1", "reply": "done", "iterations": 2, "finish_reason": "text",
                  "tool_calls": [{"name": "write_file", "error": False}], "usage": {"prompt_tokens": 100}}
        return report, 0, ""

    monkeypatch.setattr(run, "run_agent", fake_agent)
    result = run.run_task(task, [], {}, keep=False, log_dir=None)
    assert result.passed, result.failure
    assert turns == [[], ["-s", "s1"]]  # the followup continues the session
    assert result.metrics["tool_calls"] == 2 and result.metrics["prompt_tokens"] == 200


def test_compare(tmp_path: Path):
    def report(label, passed):
        runs = [{"passed": p, "metrics": {"tool_calls": 1}, "seconds": 1.0, "failure": ""} for p in passed]
        return {"label": label, "model": "m", "results": [{"task_id": "a", "runs": runs}],
                "summary": {"pass_rate": sum(passed) / len(passed), "mean_tool_calls": 1, "mean_seconds": 1,
                            "mean_prompt_tokens": 10, "by_category": {"simple": f"{sum(passed)}/{len(passed)}"}}}

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report("before", [True, False])))
    b.write_text(json.dumps(report("after", [True, True])))
    table = run.compare([a, b])
    assert "before" in table and "after" in table and "1/2" in table and "2/2" in table
