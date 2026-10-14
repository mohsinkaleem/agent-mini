#!/usr/bin/env python3
"""Agent Mini task-eval harness.

Stdlib only. For each task: copy a fixture into a temp workspace, run
``agent-mini chat -m`` headless in a throwaway home, then check the files it
left and the answer it gave. Metrics come from the run report the CLI writes
(``chat --report``), not from scraping the terminal output.

Usage
-----
    python evals/run.py                              # every task, configured model
    python evals/run.py --model qwen3.5:4b --repeat 3
    python evals/run.py --category simple            # one category
    python evals/run.py --task codebase_qa,log_tail  # some tasks
    python evals/run.py --set agent.temperature=0.2  # A/B a config change
    python evals/run.py --validate                   # check the checks (no LLM)
    python evals/run.py --compare results/a.json results/b.json

Task files (``tasks/*.toml``)
-----------------------------
    id, prompt, category, timeout_seconds
    setup            fixture directory copied into the workspace
    setup_shell      shell run in the workspace before the agent starts
    followups        more prompts, sent as later turns of the same session
    new_session      true: each followup starts a fresh session
    config           table merged into the agent config (e.g. memory)
    max_tool_calls   fail when the agent used more tool calls than this
    forbidden_tools  fail when the agent called any of these
    fixture_unchanged  paths the agent must not modify
    solution         shell that solves the task (for --validate)
    solution_answer  a correct final answer (for --validate)

    [[checks]]       ``shell`` runs in the workspace and must exit 0; without
                     ``shell`` the matchers apply to the agent's final reply.
                     Matchers: expect_exact (after strip), expect_regex,
                     expect_stdout (substring), expect_no_stdout.
                     ``{python}`` is replaced by this interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EVALS_DIR = Path(__file__).parent
TASKS_DIR = EVALS_DIR / "tasks"
RESULTS_DIR = EVALS_DIR / "results"

# Exit codes from `agent-mini chat -m` (see cli.py).
_EXIT_REASONS = {2: "max_iterations", 3: "provider_error", 4: "stuck", 124: "timeout", 130: "cancelled"}


@dataclass
class Check:
    description: str = ""
    shell: str | None = None
    expect_exact: str | None = None
    expect_regex: str | None = None
    expect_stdout: str | None = None
    expect_no_stdout: str | None = None


@dataclass
class Task:
    id: str
    prompt: str
    category: str = "general"
    timeout_seconds: int = 180
    setup: str | None = None
    setup_shell: str | None = None
    followups: list[str] = field(default_factory=list)
    new_session: bool = False
    config: dict = field(default_factory=dict)
    max_tool_calls: int | None = None
    forbidden_tools: list[str] = field(default_factory=list)
    fixture_unchanged: list[str] = field(default_factory=list)
    solution: str | None = None
    solution_answer: str | None = None
    checks: list[Check] = field(default_factory=list)


@dataclass
class Run:
    passed: bool
    failure: str
    seconds: float
    metrics: dict
    workspace: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────

_TASK_FIELDS = set(Task.__dataclass_fields__)
_CHECK_FIELDS = set(Check.__dataclass_fields__)


def parse_task(path: Path) -> Task:
    data = tomllib.loads(path.read_text())
    if "id" not in data or "prompt" not in data:
        raise ValueError(f"{path.name}: a task needs `id` and `prompt`")
    unknown = set(data) - _TASK_FIELDS
    if unknown:
        raise ValueError(f"{path.name}: unknown field(s) {sorted(unknown)}")
    checks = []
    for c in data.pop("checks", []):
        bad = set(c) - _CHECK_FIELDS
        if bad:
            raise ValueError(f"{path.name}: unknown check field(s) {sorted(bad)}")
        checks.append(Check(**c))
    data["prompt"] = data["prompt"].strip()
    return Task(**data, checks=checks)


def load_tasks(ids: list[str] | None = None, category: str | None = None) -> list[Task]:
    tasks = []
    for p in sorted(TASKS_DIR.glob("*.toml")):
        try:
            t = parse_task(p)
        except Exception as e:  # a broken task file must not hide the others
            print(f"[warn] {e}", file=sys.stderr)
            continue
        if (ids and t.id not in ids) or (category and t.category != category):
            continue
        tasks.append(t)
    return tasks


# ─────────────────────────────────────────────────────────────────────
# Workspace, home and agent runs
# ─────────────────────────────────────────────────────────────────────


def _shell(command: str, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        command.replace("{python}", shlex.quote(sys.executable)),
        shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )


def prepare_workspace(task: Task) -> Path:
    ws = Path(tempfile.mkdtemp(prefix=f"agent-mini-eval-{task.id}-"))
    if task.setup:
        shutil.copytree(EVALS_DIR / task.setup, ws, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if task.setup_shell:
        proc = _shell(task.setup_shell, ws)
        if proc.returncode:
            raise RuntimeError(f"{task.id}: setup_shell failed: {proc.stderr[:300]}")
    return ws


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out.get(key, {}), value) if isinstance(value, dict) else value
    return out


def _dotted(pairs: list[str]) -> dict:
    """``["agent.temperature=0.2"]`` → ``{"agent": {"temperature": 0.2}}`` (values parsed as JSON when possible)."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def user_provider_settings() -> dict:
    """Provider settings from the user's config; nothing else of theirs enters a run."""
    home = Path(os.environ.get("AGENT_MINI_HOME") or Path.home() / ".agent-mini").expanduser()
    try:
        cfg = json.loads((home / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        cfg = {}
    return {"provider": cfg.get("provider", "ollama"), "providers": cfg.get("providers", {})}


def prepare_home(workspace: Path, task: Task, overrides: dict) -> Path:
    """Temp ``AGENT_MINI_HOME``: the user's provider, code defaults for the rest."""
    home = Path(tempfile.mkdtemp(prefix="agent-mini-eval-home-"))
    config = {
        **user_provider_settings(),
        "tools": {"sandboxLevel": "workspace"},
        "memory": {"enabled": False},
        "workspace": str(workspace),
    }
    config = _merge(_merge(config, task.config), overrides)
    path = home / "config.json"
    path.write_text(json.dumps(config, indent=2))
    path.chmod(0o600)  # may hold API keys
    return home


def run_agent(prompt: str, ws: Path, home: Path, args: list[str], timeout: float) -> tuple[dict | None, int, str]:
    """One ``chat -m`` turn. Returns (report or None, exit code, output tail)."""
    report_path = home / f"report-{time.monotonic_ns()}.json"
    cmd = [sys.executable, "-m", "agent_mini", "chat", "-m", prompt, "--no-markdown", "--yes",
           "--report", str(report_path), *args]
    env = {**os.environ, "AGENT_MINI_HOME": str(home), "AGENT_MINI_WORKSPACE": str(ws),
           "NO_COLOR": "1", "TERM": "dumb"}
    proc = subprocess.Popen(cmd, cwd=str(home), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=max(1.0, timeout))
        code = proc.returncode
    except subprocess.TimeoutExpired:
        # Ctrl+C first: the agent then writes what it did so far into its report.
        os.killpg(proc.pid, signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
        code = 124
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return report, code, (out or "")[-3000:]


# ─────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────


def _match(check: Check, text: str) -> str:
    """Return why *text* fails the check's matchers, or ''."""
    if check.expect_exact is not None and text.strip() != check.expect_exact:
        return f"expected exactly {check.expect_exact!r}, got {text.strip()[:80]!r}"
    if check.expect_regex is not None and not re.search(check.expect_regex, text):
        return f"no match for /{check.expect_regex}/ in {text.strip()[:80]!r}"
    if check.expect_stdout is not None and check.expect_stdout not in text:
        return f"missing {check.expect_stdout!r}"
    if check.expect_no_stdout is not None and check.expect_no_stdout in text:
        return f"unexpectedly contains {check.expect_no_stdout!r}"
    return ""


def run_check(check: Check, ws: Path, reply: str) -> str:
    """'' when the check passes, otherwise the reason."""
    if check.shell is None:
        return _match(check, reply)
    try:
        proc = _shell(check.shell, ws)
    except subprocess.TimeoutExpired:
        return "check timed out"
    if proc.returncode:
        return f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:150]}"
    return _match(check, proc.stdout)


def _hash_tree(path: Path) -> str:
    """Content hash of a file or directory, ignoring caches that test runs create."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*")
        if p.is_file() and not {"__pycache__", ".pytest_cache"} & set(p.relative_to(path).parts)
    )
    for f in files:
        h.update(f.name.encode() if f == path else str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def tampered(task: Task, ws: Path) -> str:
    """The first protected path the agent modified, or ''."""
    if not task.setup:
        return ""
    src = EVALS_DIR / task.setup
    return next((rel for rel in task.fixture_unchanged if _hash_tree(src / rel) != _hash_tree(ws / rel)), "")


def judge(task: Task, ws: Path, reply: str, tool_calls: list[dict]) -> str:
    """'' when the run passes, otherwise the first reason it failed."""
    if rel := tampered(task, ws):
        return f"fixture_tampered: {rel}"
    used = sorted({c["name"] for c in tool_calls} & set(task.forbidden_tools))
    if used:
        return f"forbidden tool: {', '.join(used)}"
    if task.max_tool_calls is not None and len(tool_calls) > task.max_tool_calls:
        return f"too many tool calls: {len(tool_calls)} > {task.max_tool_calls}"
    for check in task.checks:
        if reason := run_check(check, ws, reply):
            return f"{check.description or check.shell}: {reason}"
    return ""


# ─────────────────────────────────────────────────────────────────────
# Running a task
# ─────────────────────────────────────────────────────────────────────


def _metrics(reports: list[dict]) -> dict:
    calls = [c for r in reports for c in r.get("tool_calls", [])]
    usage = [r.get("usage", {}) for r in reports]
    return {
        "iterations": sum(r.get("iterations", 0) for r in reports),
        "tool_calls": len(calls),
        "tool_errors": sum(1 for c in calls if c.get("error")),
        "tools": [c["name"] for c in calls],
        "text_tool_calls": sum(r.get("text_tool_calls", 0) for r in reports),
        "loop_nudges": sum(r.get("loop_nudges", 0) for r in reports),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "finish": [r.get("finish_reason", "?") for r in reports],
    }


def run_task(task: Task, agent_args: list[str], overrides: dict, keep: bool, log_dir: Path | None) -> Run:
    ws = prepare_workspace(task)
    home = prepare_home(ws, task, overrides)
    started = time.monotonic()
    reports: list[dict] = []
    failure, session = "", None
    for prompt in [task.prompt, *task.followups]:
        left = task.timeout_seconds - (time.monotonic() - started)
        args = [*agent_args, *(["-s", session] if session and not task.new_session else [])]
        report, code, output = run_agent(prompt, ws, home, args, left)
        if report:
            reports.append(report)
            session = report.get("session")
        if code:
            failure = _EXIT_REASONS.get(code, f"agent exited {code}: {output[-200:]}")
            break
        if report is None:
            failure = f"no report: {output[-200:]}"
            break
    seconds = time.monotonic() - started
    calls = [c for r in reports for c in r.get("tool_calls", [])]
    reply = reports[-1].get("reply", "") if reports else ""
    failure = failure or judge(task, ws, reply, calls)
    if failure and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{task.id}-{time.strftime('%H%M%S')}.json").write_text(
            json.dumps({"task": task.id, "failure": failure, "reports": reports}, indent=1, default=str)
        )
    shutil.rmtree(home, ignore_errors=True)
    if not (keep or failure):
        shutil.rmtree(ws, ignore_errors=True)
    return Run(passed=not failure, failure=failure, seconds=round(seconds, 1), metrics=_metrics(reports),
               workspace=str(ws) if failure or keep else None)


# ─────────────────────────────────────────────────────────────────────
# --validate: the checks must fail on the untouched fixture and pass on the solution
# ─────────────────────────────────────────────────────────────────────


def validate(tasks: list[Task]) -> list[str]:
    problems = []
    for task in tasks:
        ws = prepare_workspace(task)
        try:
            if not judge(task, ws, "", []):
                problems.append(f"{task.id}: checks pass before the agent has done anything")
            if task.solution is None and task.solution_answer is None:
                problems.append(f"{task.id}: no solution or solution_answer to validate against")
                continue
            if task.solution:
                proc = _shell(task.solution, ws)
                if proc.returncode:
                    problems.append(f"{task.id}: solution failed: {proc.stderr.strip()[:200]}")
                    continue
            if reason := judge(task, ws, task.solution_answer or "", []):
                problems.append(f"{task.id}: checks fail on the solution: {reason}")
        finally:
            shutil.rmtree(ws, ignore_errors=True)
    return problems


# ─────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def summarize(results: dict[str, list[Run]], tasks: dict[str, Task]) -> dict:
    runs = [r for rs in results.values() for r in rs]
    by_cat: dict[str, list[int]] = {}
    for tid, rs in results.items():
        cat = by_cat.setdefault(tasks[tid].category, [0, 0])
        cat[0] += sum(r.passed for r in rs)
        cat[1] += len(rs)
    return {
        "tasks": len(results),
        "runs": len(runs),
        "passed": sum(r.passed for r in runs),
        "pass_rate": round(sum(r.passed for r in runs) / len(runs), 3) if runs else 0.0,
        "all_runs_passed": sum(all(r.passed for r in rs) for rs in results.values()),
        "mean_seconds": _mean([r.seconds for r in runs]),
        "mean_tool_calls": _mean([r.metrics["tool_calls"] for r in runs]),
        "mean_prompt_tokens": _mean([r.metrics["prompt_tokens"] for r in runs]),
        "by_category": {c: f"{p}/{n}" for c, (p, n) in sorted(by_cat.items())},
    }


def print_table(results: dict[str, list[Run]], tasks: dict[str, Task]) -> None:
    id_w = max(map(len, results)) if results else 4
    print(f"{'TASK'.ljust(id_w)}  {'CATEGORY':<12} PASS   TIME  CALLS  TOKENS  FAILURE")
    for tid, rs in results.items():
        passed = sum(r.passed for r in rs)
        fail = next((r.failure for r in rs if not r.passed), "")
        print(
            f"{tid.ljust(id_w)}  {tasks[tid].category:<12} {passed}/{len(rs):<3} "
            f"{_mean([r.seconds for r in rs]):>5.1f}s {_mean([r.metrics['tool_calls'] for r in rs]):>5} "
            f"{_mean([r.metrics['prompt_tokens'] for r in rs]):>7.0f}  {fail[:70]}"
        )


def compare(paths: list[Path]) -> str:
    reports = [json.loads(p.read_text()) for p in paths]
    labels = [rep.get("label") or rep["model"] for rep in reports]
    w = max(12, *map(len, labels))
    ids = list(dict.fromkeys(r["task_id"] for rep in reports for r in rep["results"]))
    id_w = max(map(len, ids))
    lines = [f"{'TASK'.ljust(id_w)}  " + "  ".join(lbl.ljust(w) for lbl in labels)]
    for tid in ids:
        cells = []
        for rep in reports:
            r = next((r for r in rep["results"] if r["task_id"] == tid), None)
            if r is None:
                cells.append("--".ljust(w))
                continue
            runs = r["runs"]
            passed = sum(x["passed"] for x in runs)
            calls = _mean([x["metrics"]["tool_calls"] for x in runs])
            cells.append(f"{passed}/{len(runs)} {calls:>4}c".ljust(w))
        lines.append(f"{tid.ljust(id_w)}  " + "  ".join(cells))
    for key in ("pass_rate", "mean_tool_calls", "mean_seconds", "mean_prompt_tokens"):
        lines.append(f"{key.ljust(id_w)}  " + "  ".join(str(rep["summary"][key]).ljust(w) for rep in reports))
    cats = list(dict.fromkeys(c for rep in reports for c in rep["summary"]["by_category"]))
    for cat in cats:
        lines.append(f"  {cat.ljust(id_w - 2)}  " + "  ".join(
            rep["summary"]["by_category"].get(cat, "--").ljust(w) for rep in reports))
    return "\n".join(lines)


def save(path: Path, label: str, args: argparse.Namespace, results: dict[str, list[Run]], tasks: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "model": args.model or "(config)",
        "provider": args.provider or "(config)",
        "settings": args.set,
        "repeat": args.repeat,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summarize(results, tasks),
        "results": [
            {"task_id": tid, "category": tasks[tid].category, "runs": [r.__dict__ for r in rs]}
            for tid, rs in results.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=1))


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent Mini task-eval runner.")
    ap.add_argument("--task", help="Comma-separated task ids.")
    ap.add_argument("--category", help="Only tasks in this category.")
    ap.add_argument("--provider", help="Provider for `agent-mini chat --provider`.")
    ap.add_argument("--model", help="Model for `agent-mini chat --model`.")
    ap.add_argument("--repeat", type=int, default=1, help="Runs per task (default 1).")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Config override, e.g. agent.temperature=0.2 (repeatable).")
    ap.add_argument("--label", help="Name for this run in reports and --compare.")
    ap.add_argument("--out", help="Where to write the JSON report.")
    ap.add_argument("--keep-workspace", action="store_true", help="Keep workspaces of passing runs too.")
    ap.add_argument("--validate", action="store_true", help="Check the checks against each task's solution.")
    ap.add_argument("--compare", nargs="+", help="Compare saved reports and exit.")
    args = ap.parse_args()

    if args.compare:
        print(compare([Path(p) for p in args.compare]))
        return 0

    tasks = load_tasks(args.task.split(",") if args.task else None, args.category)
    if not tasks:
        print("No matching tasks.", file=sys.stderr)
        return 1

    if args.validate:
        problems = validate(tasks)
        for p in problems:
            print(f"✗ {p}")
        print(f"{len(tasks) - len({p.split(':')[0] for p in problems})}/{len(tasks)} tasks valid")
        return 1 if problems else 0

    agent_args = [*(["--provider", args.provider] if args.provider else []),
                  *(["--model", args.model] if args.model else [])]
    overrides = _dotted(args.set)
    label = args.label or "_".join(filter(None, [args.model or "config", *args.set]))
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", label)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else RESULTS_DIR / f"{safe}_{stamp}.json"
    log_dir = RESULTS_DIR / "failures" / f"{safe}_{stamp}"

    print(f"{len(tasks)} task(s) × {args.repeat} against {label}\n")
    by_id = {t.id: t for t in tasks}
    results: dict[str, list[Run]] = {}
    for task in tasks:
        for i in range(args.repeat):
            run = run_task(task, agent_args, overrides, args.keep_workspace, log_dir)
            results.setdefault(task.id, []).append(run)
            mark = "PASS" if run.passed else f"FAIL  {run.failure[:90]}"
            print(f"  {task.id} [{i + 1}/{args.repeat}] {run.seconds:5.1f}s {run.metrics['tool_calls']:>2}c  {mark}",
                  flush=True)
        save(out, label, args, results, by_id)  # partial results survive an interrupted run

    print()
    print_table(results, by_id)
    s = summarize(results, by_id)
    print(f"\npass rate {s['pass_rate']:.0%} ({s['passed']}/{s['runs']} runs) · "
          f"{s['all_runs_passed']}/{s['tasks']} tasks passed every run · {s['by_category']}")
    print(f"report → {out}")
    return 0 if s["passed"] == s["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
