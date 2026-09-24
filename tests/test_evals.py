"""Tests for the eval harness itself: task parsing and check correctness (no LLM)."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RUN_PY = Path(__file__).resolve().parent.parent / "evals" / "run.py"
_spec = importlib.util.spec_from_file_location("agent_mini_evals_run", _RUN_PY)
run = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = run
_spec.loader.exec_module(run)


def _task(task_id: str):
    (task,) = run.load_tasks(only=task_id)
    return task


def _prepare(task, tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(run, "tempfile", SimpleNamespace(mkdtemp=lambda prefix="": str(tmp_path)))
    return run._prepare_workspace(task)


def _failed_checks(task, workspace: Path) -> list[str]:
    return [c.description for c in task.checks if not run._run_check(c, workspace)[0]]


def test_all_tasks_parse():
    tasks = run.load_tasks()
    assert len(tasks) == len(list(run.TASKS_DIR.glob("*.toml")))
    for task in tasks:
        assert task.checks, f"{task.id} has no checks"


def test_expect_exact_rejects_substring(tmp_path: Path):
    check = run.Check(shell="echo 120", expect_exact="20")
    assert not run._run_check(check, tmp_path)[0]
    assert run._run_check(run.Check(shell="echo 20", expect_exact="20"), tmp_path)[0]


def test_python_placeholder_uses_harness_interpreter(tmp_path: Path):
    check = run.Check(shell="{python} -c 'import sys; print(sys.executable)'", expect_exact=sys.executable)
    assert run._run_check(check, tmp_path)[0]


def test_agent_metrics_parse_turn_footer():
    out = (
        "Recovered 1 tool call(s) written as text\n"
        "Loop detected: nudging the model\n"
        "reached max iterations (15)\n"
        "  7 iterations  •  12,345 in  •  678 out  •  9.1s\n"
    )
    m = run._agent_metrics(out)
    assert m == {
        "iterations": 7, "json_repairs": 0, "text_tool_calls": 1, "loop_nudges": 1, "tokens": 13023,
    }


def test_write_file_from_scratch_checks(tmp_path: Path):
    """E3: the greet() check must pass for a correct solution and fail without one."""
    task = _task("write_file_from_scratch")
    assert _failed_checks(task, tmp_path)

    (tmp_path / "hello.py").write_text('def greet(name):\n    return "Hello, " + name\n')
    assert _failed_checks(task, tmp_path) == []

    (tmp_path / "hello.py").write_text(
        'def greet(name):\n    return "Hello, " + name\n\n\n'
        'if __name__ == "__main__":\n    print(greet("x"))\n'
    )
    assert _failed_checks(task, tmp_path) == ["no print / __main__ block"]


def test_refactor_rename_checks(tmp_path: Path, monkeypatch):
    task = _task("refactor_rename")
    ws = _prepare(task, tmp_path, monkeypatch)
    assert _failed_checks(task, ws)

    for rel in ("src/mymath.py", "tests/test_mymath.py"):
        path = ws / rel
        path.write_text(re.sub(r"\bcalc\b", "compute", path.read_text()))
    assert _failed_checks(task, ws) == []


@pytest.mark.parametrize("tamper", [False, True])
def test_fixture_unchanged_detects_edits(tmp_path: Path, monkeypatch, tamper: bool):
    task = _task("fix_failing_test")
    ws = _prepare(task, tmp_path, monkeypatch)
    (ws / "tests" / "__pycache__").mkdir(exist_ok=True)
    (ws / "tests" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    if tamper:
        test_file = ws / "tests" / "test_stack.py"
        test_file.write_text(test_file.read_text().replace("== 3", "== 1"))
    assert run._check_fixture_unchanged(task, ws) == ("tests" if tamper else "")


def test_eval_home_keeps_provider_settings_only(tmp_path: Path, monkeypatch):
    """E5: runs get the user's provider config but not their memory or prompt."""
    user_home = tmp_path / "user"
    user_home.mkdir()
    (user_home / "config.json").write_text(json.dumps({
        "provider": "openai",
        "providers": {"openai": {"apiKey": "sk-x", "model": "gpt-5"}},
        "agent": {"systemPrompt": "talk like a pirate", "temperature": 0.2, "maxIterations": 99},
        "memory": {"enabled": True},
        "tools": {"confirm": ["shell_exec"]},
    }))
    monkeypatch.setenv("AGENT_MINI_HOME", str(user_home))
    home = run._prepare_home(tmp_path / "ws")
    config = json.loads((home / "config.json").read_text())
    assert config["providers"]["openai"]["model"] == "gpt-5"
    assert config["agent"] == {"temperature": 0.2}
    assert config["memory"] == {"enabled": False}
    assert "confirm" not in config["tools"]
    assert config["workspace"] == str(tmp_path / "ws")
    assert (home / "config.json").stat().st_mode & 0o777 == 0o600
