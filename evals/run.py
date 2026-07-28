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
    python evals/run.py --out results/small.json  # save JSON report
    python evals/run.py --compare results/*.json  # cross-tier table

Design
------
- **Fixtures** are copied per-run to ``tempfile.mkdtemp()`` — never mutated in
  place, and always cleaned up on exit.
- **Task file format** is a flat subset of YAML parsed by ``_parse_task`` (no
  PyYAML dep). Only strings, ints, and lists of dicts are supported — that's
  all we need.
- **Checks** are shell one-liners executed with ``cwd=workspace``. Non-zero
  exit ⇒ failed check. Explicit ``expect_stdout`` / ``expect_no_stdout``
  matchers are also supported for text assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

EVALS_DIR = Path(__file__).parent
TASKS_DIR = EVALS_DIR / "tasks"
FIXTURES_DIR = EVALS_DIR / "fixtures"
RESULTS_DIR = EVALS_DIR / "results"

# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Check:
    """A single shell / stdout check applied after the agent finishes."""
    shell: str
    description: str = ""
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


# ─────────────────────────────────────────────────────────────────────
# Tiny YAML subset parser
# ─────────────────────────────────────────────────────────────────────
#
# We only support the fixed schema below — no anchors, no flow style, no
# nested dicts beyond one level of list-of-dicts. Keeping this in-tree
# means the eval harness has ZERO third-party deps.
#
# Grammar (informal):
#   file        = header block-scalar? key-value* checks-list?
#   key-value   = KEY ":" (INLINE | BLOCK)
#   BLOCK       = "|" NL indented-lines
#   checks-list = "checks:" NL ( "-" (kv-line NL)+ )*


def _parse_task(path: Path) -> Task:
    text = path.read_text()
    lines = text.splitlines()
    i = 0
    data: dict = {}
    n = len(lines)

    def _strip_quotes(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            return v[1:-1]
        return v

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Top-level list block: "checks:"
        if stripped == "checks:":
            i += 1
            checks: list[dict] = []
            current: dict | None = None
            while i < n:
                line_inner = lines[i]
                if line_inner.strip() == "" or line_inner.strip().startswith("#"):
                    i += 1
                    continue
                # Detect end of block: dedent to column 0 with content
                if line_inner and not line_inner.startswith(" ") and not line_inner.startswith("\t"):
                    break
                s = line_inner.strip()
                if s.startswith("- "):
                    if current is not None:
                        checks.append(current)
                    current = {}
                    kv = s[2:]
                    if ":" in kv:
                        k, _, v = kv.partition(":")
                        current[k.strip()] = _strip_quotes(v)
                elif current is not None and ":" in s:
                    k, _, v = s.partition(":")
                    current[k.strip()] = _strip_quotes(v)
                i += 1
            if current is not None:
                checks.append(current)
            data["checks"] = checks
            continue

        # Block scalar: "prompt: |"
        if stripped.endswith(": |"):
            key = stripped[:-3].strip()
            i += 1
            block: list[str] = []
            base_indent: int | None = None
            while i < n:
                line_inner = lines[i]
                if line_inner.strip() == "":
                    block.append("")
                    i += 1
                    continue
                indent = len(line_inner) - len(line_inner.lstrip(" "))
                if base_indent is None:
                    base_indent = indent
                if indent < (base_indent or 1) and line_inner.strip():
                    break
                block.append(line_inner[base_indent:] if base_indent else line_inner)
                i += 1
            data[key] = "\n".join(block).rstrip()
            continue

        # Simple "key: value"
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            data[k.strip()] = _strip_quotes(v)
        i += 1

    if "id" not in data or "prompt" not in data:
        raise ValueError(f"{path}: task requires `id` and `prompt`")

    checks = []
    for c in data.get("checks", []) or []:
        checks.append(
            Check(
                shell=c.get("shell", ""),
                description=c.get("description", ""),
                expect_stdout=c.get("expect_stdout"),
                expect_no_stdout=c.get("expect_no_stdout"),
            )
        )

    return Task(
        id=str(data["id"]),
        prompt=str(data["prompt"]),
        category=str(data.get("category", "general")),
        setup=(str(data["setup"]) if data.get("setup") else None),
        timeout_seconds=int(data.get("timeout_seconds") or 180),
        checks=checks,
    )


def load_tasks(only: str | None = None) -> list[Task]:
    if not TASKS_DIR.exists():
        return []
    tasks = []
    for p in sorted(TASKS_DIR.glob("*.yaml")):
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


def _run_agent(prompt: str, workspace: Path, model: str | None, timeout: int) -> tuple[str, int]:
    """Invoke ``agent-mini chat -m <prompt> --workspace <ws>``.

    Returns (combined stdout+stderr, exit code). ``AGENT_MINI_WORKSPACE`` is
    set so any child process inherits the pinned workspace too.
    """
    env = os.environ.copy()
    env["AGENT_MINI_WORKSPACE"] = str(workspace)
    # Force color/markdown off for machine-parseable stdout.
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"

    cmd = [
        sys.executable, "-m", "agent_mini", "chat",
        "-m", prompt,
        "--workspace", str(workspace),
        "--no-markdown",
    ]
    # Optional per-run model override via slash-command-style "-m/--model"
    # would be nicer, but keep the CLI stable and use config for now unless
    # the caller explicitly points to a Python override. We honour the
    # --model flag by writing it into the env for the (optional) config
    # loader to see; here we take the simpler route and just log it.
    if model:
        env["AGENT_MINI_MODEL"] = model  # cooperative — consumed by callers who care

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


def _run_check(check: Check, workspace: Path) -> tuple[bool, str]:
    """Execute a shell check inside *workspace*. Returns (passed, detail)."""
    try:
        proc = subprocess.run(
            check.shell,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "check timed out after 60s"

    stdout = proc.stdout or ""
    if check.expect_stdout is not None:
        if check.expect_stdout not in stdout:
            return False, f"stdout missing '{check.expect_stdout}'"
    if check.expect_no_stdout is not None:
        if check.expect_no_stdout in stdout:
            return False, f"stdout unexpectedly contains '{check.expect_no_stdout}'"
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {(proc.stderr or stdout)[:200]}"
    return True, ""


def run_task(task: Task, model: str | None = None, keep_workspace: bool = False) -> Result:
    ws = _prepare_workspace(task)
    t0 = time.monotonic()
    agent_out, agent_exit = _run_agent(task.prompt, ws, model, task.timeout_seconds)
    duration = time.monotonic() - t0

    checks_passed = 0
    failure_reason = ""
    if agent_exit == 124:
        failure_reason = "agent timed out"
    elif agent_exit != 0:
        failure_reason = f"agent exited {agent_exit}"

    for c in task.checks:
        ok, detail = _run_check(c, ws)
        if ok:
            checks_passed += 1
        elif not failure_reason:
            failure_reason = f"{c.description or c.shell}: {detail}"

    passed = agent_exit == 0 and checks_passed == len(task.checks)

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
        f"{'RESULT':<7}  {'CHECKS':<9}  {'TIME':>7}  DETAIL"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        checks = f"{r.checks_passed}/{r.checks_total}"
        detail = "" if r.passed else (r.failure_reason[:60] or "")
        rows.append(
            f"{r.task_id.ljust(id_w)}  {r.category.ljust(cat_w)}  "
            f"{status:<7}  {checks:<9}  {r.duration_seconds:>6.1f}s  {detail}"
        )
    pass_n = sum(1 for r in results if r.passed)
    rows.append("-" * len(header))
    rows.append(f"SUMMARY: {pass_n}/{len(results)} tasks passed")
    return "\n".join(rows)


def _agent_metrics(agent_stdout: str) -> dict:
    """Extract lightweight metrics from the agent's stderr log."""
    def _count(pattern: str) -> int:
        return len(re.findall(pattern, agent_stdout))

    tokens_match = re.search(r"(\d+)→\s*(\d+)←\s*\((\d+)\s*total\)", agent_stdout)
    tokens = int(tokens_match.group(3)) if tokens_match else 0

    return {
        "iterations": _count(r"iteration \d+"),
        "json_repairs": _count(r"repair|Repaired"),
        "summarize_hits": _count(r"summariz"),
        "loop_nudges": _count(r"repeated the same tool call"),
        "tokens": tokens,
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
                "metrics": _agent_metrics(r.agent_stdout),
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
    ap.add_argument("--model", help="Model name (informational; requires config to point here).")
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

    model = args.model or os.environ.get("AGENT_MINI_MODEL") or "(config)"
    print(f"Running {len(tasks)} task(s) against model={model}\n")

    results: list[Result] = []
    for t in tasks:
        print(f"→ {t.id} ({t.category}) ", end="", flush=True)
        r = run_task(t, model=args.model, keep_workspace=args.keep_workspace)
        results.append(r)
        print(f"[{ 'PASS' if r.passed else 'FAIL' }] {r.duration_seconds:.1f}s")

    print()
    print(_format_table(results))

    out_path = _save_report(model, results, Path(args.out) if args.out else None)
    print(f"\nreport → {out_path}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
