# Agent Mini — Task Evals

A tiny, framework-free harness that measures whether the agent can complete
**real-world tasks** end-to-end, not just pass unit tests. This is where the
"small-model optimized" claim gets tested for real.

Zero new dependencies. Uses only the standard library + `agent-mini` itself.

---

## Quick start

```bash
# 1. Have Ollama (or any provider) already running with a model
ollama pull llama3.1:8b

# 2. Point agent-mini at it (only needed once)
agent-mini init

# 3. Run the whole suite
python evals/run.py

# 4. Run against a specific model
python evals/run.py --model llama3.2:3b

# 5. Run a single task
python evals/run.py --task refactor_rename

# 6. Compare tiers (the thesis test)
python evals/run.py --model llama3.2:3b   --out results/tiny.json
python evals/run.py --model llama3.1:8b   --out results/small.json
python evals/run.py --model qwen2.5:14b   --out results/medium.json
python evals/run.py --compare results/*.json
```

Results are printed as a table and written to `results/<model>_<timestamp>.json`
(one line per task, so you can `jq` / `grep` them).

---

## Latest results — qwen3.5 tiers (Ollama, 2026-07-28)

Local M-series Mac, `think: false`, default agent config (`maxIterations: 20`).

| Task                       | qwen3.5:4b   | qwen3.5:9b   | qwen3.5:27b   |
|----------------------------|--------------|--------------|---------------|
| codebase_qa                | PASS · 18.0s | PASS · 23.8s | PASS · 81.5s  |
| error_recovery             | PASS ·  8.7s | PASS · 19.5s | PASS · 50.3s  |
| fix_failing_test           | PASS · 17.9s | PASS · 24.9s | PASS · 85.3s  |
| list_directory_basic       | FAIL ·  8.3s | PASS · 14.0s | PASS · 47.5s  |
| multi_file_edit            | PASS ·  9.5s | PASS · 18.3s | PASS · 78.4s  |
| refactor_rename            | FAIL · 10.1s | FAIL · 61.0s | PASS · 184.9s |
| search_and_summarize       | PASS ·  6.2s | PASS · 23.2s | PASS · 32.0s  |
| shell_pipeline             | FAIL · 12.6s | PASS · 32.2s | FAIL · 120.0s (timeout) |
| write_file_from_scratch    | FAIL · 11.1s | FAIL · 15.7s | PASS · 34.7s  |
| **Pass rate**              | **5/9**      | **7/9**      | **8/9**       |

Notes:

- `qwen3.5:9b` is the sweet spot on this suite
  `shell_pipeline` and finishing every task well under the timeout.
- `write_file_from_scratch` also fails everywhere: models emit a `print`
  / `__main__` block despite the prompt forbidding it. Good candidate for
  a small-model instruction-tuning improvement.
---

## Layout

```
evals/
├── README.md          # this file
├── run.py             # ~200 lines: copy fixture → run agent headless → score
├── tasks/*.yaml       # task fixtures — prompt + programmatic check
└── fixtures/          # tiny sample repos used by tasks
```

Each task is a self-contained YAML file:

```yaml
id: refactor_rename
category: refactor
setup: fixtures/py_project
prompt: |
  Rename the function `calc` to `compute` across all .py files in this
  workspace and make sure the tests still pass.
timeout_seconds: 180
checks:
  - shell: "! grep -rq 'def calc(' ."
    description: "No `def calc` remains"
  - shell: "grep -rq 'def compute(' ."
    description: "`def compute` appears somewhere"
  - shell: "python -m pytest -q"
    description: "Tests still pass"
```

Add a task: drop a YAML file in `tasks/` and (optionally) a matching
fixture directory in `fixtures/`. The runner picks it up on next run.

---

## Task categories (the surface we actually care about)

| Category | Exercises | Task ID |
|----------|-----------|---------|
| File ops | `write_file`, `code_edit`, multi-file rename | `refactor_rename` |
| Codebase Q&A | `search_files` → `read_file` → summarize | `codebase_qa` |
| Multi-step | write, run, verify a fix | `fix_failing_test` |
| Error recovery | retry a different way after failure | `error_recovery` |
| Memory | store fact turn 1, recall turn 3 (uses `--multi-turn`) | `memory_recall` |
| Search | grep + read | `search_and_summarize` |

---

## Metrics (per model)

The runner tabulates:

| Metric | Why it matters |
|--------|----------------|
| Success rate | Core efficacy. |
| Median iterations | Loop efficiency. Wandering = bad. |
| JSON-repair fire rate | Is small-model tool-call repair earning its keep? |
| Summarize trigger rate | Is context management engaging when needed? |
| Loop-detection hits | Are small models stalling? |
| Total tokens / wall time | Cost. |

The **thesis test**: run the same suite across `tiny` / `small` / `medium` /
`cloud` tiers. If tuned `3B` ≈ untuned `8B`, the optimization is proven. If
not, it's theater — and this harness is exactly what surfaces that.

---

## Design constraints

- **No new dependencies.** stdlib only (`subprocess`, `shutil`, `tempfile`,
  `json`, `re`, `argparse`). YAML is parsed by a tiny in-file loader — the
  task schema is intentionally flat enough not to need PyYAML.
- **No mocking.** Real provider, real tools, real filesystem. Fixtures are
  copied to a fresh `tempfile.mkdtemp()` per task, so runs don't leak.
- **Zero framework bloat.** ~200 LOC in `run.py`. Read it in five minutes.
- **CI-friendly.** Gate as nightly against a live model; keep unit tests as
  the fast job. Model latency should not block PRs.

## Adding a task

1. Create `evals/fixtures/my_fixture/` with a minimal set of files.
2. Create `evals/tasks/my_task.yaml`:
   - `id`: matches the file stem.
   - `setup`: path to the fixture (relative to `evals/`).
   - `prompt`: what to ask the agent.
   - `checks`: shell one-liners that must exit `0`.
3. Verify: `python evals/run.py --task my_task`.

Each bug we fix should get a new task. The corpus grows organically.
