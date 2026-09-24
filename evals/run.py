#!/usr/bin/env python3
"""Agent Mini task-eval harness.

Framework-free: stdlib + `agent-mini` only. No YAML lib, no pytest, no
LangChain. Copies a fixture to a temp workspace, runs the agent headless
against the prompt, then runs shell checks to verify the outcome.

Usage
-----
    python evals/run.py                            # all tasks, config model
    python evals/run.py --task refactor_rename    # one task
    python evals/run.py --model llama3.2:3b       # override model
    python evals/run.py --provider ollama --model qwen3:8b
    python evals/run.py --out results/small.json  # save JSON report
    python evals/run.py --compare results/*.json  # cross-tier table

Design
------
- **Fixtures** are copied per-run to ``tempfile.mkdtemp()`` — never mutated in
  place, and always cleaned up on exit.
- **Task files** are TOML, parsed with stdlib ``tomllib``.
- **Checks** are shell one-liners executed with ``cwd=workspace``. Non-zero
  exit ⇒ failed check. ``{python}`` is replaced by the harness interpreter.
  Matchers: ``expect_exact`` (stripped stdout equals), ``expect_regex``,
  ``expect_stdout`` (substring), ``expect_no_stdout``. A task can also list
  ``fixture_unchanged`` paths that the agent must not modify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EVALS_DIR = Path(__file__).parent
TASKS_DIR = EVALS_DIR / "tasks"
FIXTURES_DIR = EVALS_DIR / "fixtures"
RESULTS_DIR = EVALS_DIR / "results"

# Exit codes from `agent-mini chat -m` (see cli.py).
_AGENT_EXIT_REASONS = {
    2: "max_iterations",
    3: "provider_error",
    4: "stuck",
    124: "timeout",
    130: "cancelled",
}

# Agent settings copied from the user's config into the eval home; everything
# else (system prompt, memory, plugins, sessions) stays out of the run.
_COPIED_AGENT_KEYS = ("temperature", "tier", "contextWindow")

# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Check:
    """A single shell / stdout check applied after the agent finishes."""
    shell: str
    description: str = ""
    expect_exact: str | None = None
    expect_regex: str | None = None
    expect_stdout: str | None = None
    expect_no_stdout: str | None = None


@dataclass
class Task:
    id: str
    prompt: str
    category: str = "general"
    setup: str | None = None
    timeout_seconds: int = 180
    checks: list[Check] = field(default_factory=list)
    fixture_unchanged: list[str] = field(default_factory=list)


@dataclass
class Result:
    task_id: str
    category: str
    passed: bool
    duration_seconds: float
    checks_passed: int
    checks_total: int
    agent_stdout: str
    workspace: str
    failure_reason: str = ""
    metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Task loading (TOML via stdlib tomllib)
# ─────────────────────────────────────────────────────────────────────

_CHECK_FIELDS = set(Check.__dataclass_fields__)


def _parse_task(path: Path) -> Task:
    data = tomllib.loads(path.read_text())
    if "id" not in data or "prompt" not in data:
        raise ValueError(f"{path}: task requires `id` and `prompt`")

    checks = []
    for c in data.get("checks", []):
        unknown = set(c) - _CHECK_FIELDS
        if unknown:
            raise ValueError(f"{path}: unknown check field(s) {sorted(unknown)}")
        checks.append(Check(**c))

    return Task(
        id=str(data["id"]),
        prompt=str(data["prompt"]).strip(),
        category=str(data.get("category", "general")),
        setup=data.get("setup") or None,
        timeout_seconds=int(data.get("timeout_seconds", 180)),
        checks=checks,
        fixture_unchanged=list(data.get("fixture_unchanged", [])),
    )


def load_tasks(only: str | None = None) -> list[Task]:
    if not TASKS_DIR.exists():
        return []
    tasks = []
    for p in sorted(TASKS_DIR.glob("*.toml")):
        try:
            t = _parse_task(p)
        except Exception as e:
            print(f"[warn] failed to parse {p.name}: {e}", file=sys.stderr)
            continue
        if only and t.id != only:
            continue
        tasks.append(t)
    return tasks


# ─────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────


def _prepare_workspace(task: Task) -> Path:
    ws = Path(tempfile.mkdtemp(prefix=f"agent-mini-eval-{task.id}-"))
    if task.setup:
        src = (EVALS_DIR / task.setup).resolve()
        if not src.exists():
            raise FileNotFoundError(f"fixture not found: {src}")
        # Copy fixture contents into workspace (not a nested dir).
        for entry in src.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, ws / entry.name)
            else:
                shutil.copy2(entry, ws / entry.name)
    return ws


def _prepare_home(workspace: Path) -> Path:
    """Temp ``AGENT_MINI_HOME`` holding only the user's provider settings.

    Keeps the user's memory, plugins, sessions and system prompt out of the
    run, and keeps the run out of them.
    """
    home = Path(tempfile.mkdtemp(prefix="agent-mini-eval-home-"))
    user_home = Path(os.environ.get("AGENT_MINI_HOME") or Path.home() / ".agent-mini").expanduser()
    try:
        user_cfg = json.loads((user_home / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        user_cfg = {}
    agent_cfg = user_cfg.get("agent", {})
    config = {
        "provider": user_cfg.get("provider", "ollama"),
        "providers": user_cfg.get("providers", {}),
        "agent": {k: agent_cfg[k] for k in _COPIED_AGENT_KEYS if k in agent_cfg},
        "tools": {"sandboxLevel": "workspace"},
        "memory": {"enabled": False},
        "workspace": str(workspace),
    }
    path = home / "config.json"
    path.write_text(json.dumps(config, indent=2))
    path.chmod(0o600)  # may hold API keys
    return home


def _run_agent(
    prompt: str,
    workspace: Path,
    provider: str | None,
    model: str | None,
    timeout: int,
) -> tuple[str, int]:
    """Invoke ``agent-mini chat -m <prompt> --workspace <ws> --yes``.

    Returns (combined stdout+stderr, exit code). ``AGENT_MINI_WORKSPACE`` is
    set so any child process inherits the pinned workspace too, and
    ``AGENT_MINI_HOME`` points at a throwaway home (see ``_prepare_home``).
    """
    home = _prepare_home(workspace)
    env = os.environ.copy()
    env["AGENT_MINI_WORKSPACE"] = str(workspace)
    env["AGENT_MINI_HOME"] = str(home)
    # Force color/markdown off for machine-parseable stdout.
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"

    cmd = [
        sys.executable, "-m", "agent_mini", "chat",
        "-m", prompt,
        "--workspace", str(workspace),
        "--no-markdown",
        "--yes",
    ]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out, proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return f"[TIMEOUT after {timeout}s]\n{out}", 124
    finally:
        shutil.rmtree(home, ignore_errors=True)


def _run_check(check: Check, workspace: Path) -> tuple[bool, str]:
    """Execute a shell check inside *workspace*. Returns (passed, detail)."""
    command = check.shell.replace("{python}", shlex.quote(sys.executable))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "check timed out after 60s"

    stdout = proc.stdout or ""
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {(proc.stderr or stdout)[:200]}"
    if check.expect_exact is not None and stdout.strip() != check.expect_exact:
        return False, f"expected exactly {check.expect_exact!r}, got {stdout.strip()[:100]!r}"
    if check.expect_regex is not None and not re.search(check.expect_regex, stdout):
        return False, f"stdout does not match /{check.expect_regex}/"
    if check.expect_stdout is not None and check.expect_stdout not in stdout:
        return False, f"stdout missing '{check.expect_stdout}'"
    if check.expect_no_stdout is not None and check.expect_no_stdout in stdout:
        return False, f"stdout unexpectedly contains '{check.expect_no_stdout}'"
    return True, ""


_HASH_SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def _hash_tree(path: Path) -> str:
    """Content hash of a file or directory, ignoring caches that test runs create."""
    h = hashlib.sha256()
    if not path.exists():
        return "missing"
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*")
        if p.is_file() and not _HASH_SKIP_DIRS.intersection(p.relative_to(path).parts)
    )
    for f in files:
        h.update(str(f.relative_to(path) if f != path else f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _check_fixture_unchanged(task: Task, workspace: Path) -> str:
    """Return the first protected path the agent modified, or ''."""
    if not task.setup:
        return ""
    src = (EVALS_DIR / task.setup).resolve()
    for rel in task.fixture_unchanged:
        if _hash_tree(src / rel) != _hash_tree(workspace / rel):
            return rel
    return ""


def run_task(
    task: Task,
    provider: str | None = None,
    model: str | None = None,
    keep_workspace: bool = False,
) -> Result:
    ws = _prepare_workspace(task)
    t0 = time.monotonic()
    agent_out, agent_exit = _run_agent(task.prompt, ws, provider, model, task.timeout_seconds)
    duration = time.monotonic() - t0

    checks_passed = 0
    failure_reason = ""
    if agent_exit != 0:
        failure_reason = _AGENT_EXIT_REASONS.get(agent_exit, f"agent exited {agent_exit}")

    tampered = _check_fixture_unchanged(task, ws)
    if tampered and not failure_reason:
        failure_reason = f"fixture_tampered: {tampered}"

    for c in task.checks:
        ok, detail = _run_check(c, ws)
        if ok:
            checks_passed += 1
        elif not failure_reason:
            failure_reason = f"{c.description or c.shell}: {detail}"

    passed = agent_exit == 0 and not tampered and checks_passed == len(task.checks)

    result = Result(
        task_id=task.id,
        category=task.category,
        passed=passed,
        duration_seconds=round(duration, 2),
        checks_passed=checks_passed,
        checks_total=len(task.checks),
        agent_stdout=agent_out[-4000:],  # trim to keep JSON small
        workspace=str(ws),
        failure_reason=failure_reason,
        metrics=_agent_metrics(agent_out),
    )

    if not keep_workspace and not failure_reason:
        # Only nuke the workspace if the task passed; keep on failure so
        # humans can inspect. Failed workspaces stay on disk under /tmp
        # and get GC'd by the OS.
        shutil.rmtree(ws, ignore_errors=True)
    return result


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────


def _format_table(results: list[Result]) -> str:
    if not results:
        return "(no results)"
    id_w = max(len(r.task_id) for r in results)
    cat_w = max(len(r.category) for r in results)
    rows = []
    header = (
        f"{'TASK'.ljust(id_w)}  {'CAT'.ljust(cat_w)}  "
        f"{'RESULT':<7}  {'CHECKS':<9}  {'TIME':>7}  {'ITER':>4}  {'TOKENS':>7}  DETAIL"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        checks = f"{r.checks_passed}/{r.checks_total}"
        detail = "" if r.passed else (r.failure_reason[:60] or "")
        rows.append(
            f"{r.task_id.ljust(id_w)}  {r.category.ljust(cat_w)}  "
            f"{status:<7}  {checks:<9}  {r.duration_seconds:>6.1f}s  "
            f"{r.metrics.get('iterations', 0):>4}  {r.metrics.get('tokens', 0):>7,}  {detail}"
        )
    pass_n = sum(1 for r in results if r.passed)
    rows.append("-" * len(header))
    rows.append(f"SUMMARY: {pass_n}/{len(results)} tasks passed")
    return "\n".join(rows)


def _agent_metrics(agent_stdout: str) -> dict:
    """Extract lightweight metrics from the agent's log and turn footer."""
    def _count(pattern: str) -> int:
        return len(re.findall(pattern, agent_stdout))

    def _last_int(pattern: str) -> int:
        found = re.findall(pattern, agent_stdout)
        return int(found[-1].replace(",", "")) if found else 0

    # Footer from `chat -m`: "N iterations  •  X in  •  Y out  •  T s"
    return {
        "iterations": _last_int(r"(\d+) iterations?\s+•"),
        "json_repairs": _count(r"Repaired malformed tool arguments"),
        "text_tool_calls": _count(r"Recovered \d+ tool call"),
        "loop_nudges": _count(r"Loop detected"),
        "tokens": _last_int(r"([\d,]+) in\s+•") + _last_int(r"([\d,]+) out\s+•"),
    }


def _save_report(model: str, results: list[Result], out_path: Path | None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        out_path = RESULTS_DIR / f"{safe_model}_{stamp}.json"

    payload = {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
        },
        "results": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "passed": r.passed,
                "duration_seconds": r.duration_seconds,
                "checks_passed": r.checks_passed,
                "checks_total": r.checks_total,
                "failure_reason": r.failure_reason,
                "metrics": r.metrics,
                "workspace": r.workspace if not r.passed else None,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _compare_reports(paths: list[Path]) -> str:
    reports = [json.loads(p.read_text()) for p in paths]
    if not reports:
        return "(no reports)"
    # Union of task ids across reports.
    ids: list[str] = []
    for rep in reports:
        for r in rep["results"]:
            if r["task_id"] not in ids:
                ids.append(r["task_id"])

    model_w = max(len(rep["model"]) for rep in reports)
    id_w = max(len(t) for t in ids)
    rows = []
    header = f"{'TASK'.ljust(id_w)}  " + "  ".join(rep["model"].ljust(model_w) for rep in reports)
    rows.append(header)
    rows.append("-" * len(header))
    for tid in ids:
        cells = []
        for rep in reports:
            match = next((r for r in rep["results"] if r["task_id"] == tid), None)
            if match is None:
                cells.append("--".ljust(model_w))
            else:
                mark = "PASS" if match["passed"] else "FAIL"
                cells.append(f"{mark} {match['duration_seconds']:>5.1f}s".ljust(model_w))
        rows.append(f"{tid.ljust(id_w)}  " + "  ".join(cells))
    rows.append("-" * len(header))
    summary = "  ".join(
        f"{rep['summary']['passed']}/{rep['summary']['total']}".ljust(model_w)
        for rep in reports
    )
    rows.append(f"{'SUMMARY'.ljust(id_w)}  {summary}")
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent Mini task-eval runner.")
    ap.add_argument("--task", help="Run only this task id.")
    ap.add_argument("--provider", help="Provider override passed to `agent-mini chat --provider`.")
    ap.add_argument("--model", help="Model override passed to `agent-mini chat --model`.")
    ap.add_argument("--out", help="Path to write JSON report.")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="Keep the temp workspace even when a task passes (for debugging).")
    ap.add_argument("--compare", nargs="+",
                    help="Print a cross-report comparison table and exit.")
    args = ap.parse_args()

    if args.compare:
        print(_compare_reports([Path(p) for p in args.compare]))
        return 0

    tasks = load_tasks(only=args.task)
    if not tasks:
        print(f"No tasks found in {TASKS_DIR}", file=sys.stderr)
        return 1

    model = args.model or "(config)"
    if args.provider:
        model = f"{args.provider}/{model}"
    print(f"Running {len(tasks)} task(s) against model={model}\n")

    results: list[Result] = []
    for t in tasks:
        print(f"→ {t.id} ({t.category}) ", end="", flush=True)
        r = run_task(
            t, provider=args.provider, model=args.model, keep_workspace=args.keep_workspace
        )
        results.append(r)
        print(f"[{ 'PASS' if r.passed else 'FAIL' }] {r.duration_seconds:.1f}s")

    print()
    print(_format_table(results))

    out_path = _save_report(model, results, Path(args.out) if args.out else None)
    print(f"\nreport → {out_path}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
