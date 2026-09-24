# Agent Mini — Task Evals

A tiny, framework-free harness that measures whether the agent can complete
**real-world tasks** end-to-end, not just pass unit tests. This is where the
"small-model optimized" claim gets tested for real.

Zero new dependencies. Uses only the standard library + `agent-mini` itself.

---

## Quick start

```bash
# 1. Have Ollama (or any provider) already running with a model
ollama pull qwen3:8b

# 2. Point agent-mini at it (only needed once), then check the setup
agent-mini init
agent-mini doctor

# 3. Run the whole suite
python evals/run.py

# 4. Run against a specific model (passed to `agent-mini chat --model`)
python evals/run.py --model qwen3:4b
python evals/run.py --provider openai --model gpt-5-mini

# 5. Run a single task
python evals/run.py --task refactor_rename

# 6. Compare tiers (the thesis test)
python evals/run.py --model qwen3:1.7b  --out results/tiny.json
python evals/run.py --model qwen3:8b    --out results/small.json
python evals/run.py --model qwen3:14b   --out results/medium.json
python evals/run.py --compare results/*.json
```

Results are printed as a table and written to `results/<model>_<timestamp>.json`
(one line per task, so you can `jq` / `grep` them).

---

## Latest results — qwen3.5 (Ollama, 2026-09-24)

Local M-series Mac, tier-default budgets (`small`: 15 iterations),
`temperature: 0.7`, one run per model. Cell = result · wall time ·
iterations · tokens (prompt + completion, summed over the turn).

| Task                       | qwen3.5:4b              | qwen3.5:9b              |
|----------------------------|-------------------------|-------------------------|
| codebase_qa                | PASS ·  6.6s · 4 ·  7.0k | PASS · 34.5s · 5 ·  9.0k |
| error_recovery             | PASS ·  6.6s · 4 ·  7.2k | PASS · 10.5s · 4 ·  7.2k |
| fix_failing_test           | PASS · 13.1s · 6 · 12.9k | PASS · 25.0s · 8 · 17.6k |
| list_directory_basic       | PASS ·  5.7s · 3 ·  5.1k | PASS ·  8.7s · 3 ·  5.1k |
| multi_file_edit            | PASS ·  7.7s · 4 ·  7.2k | PASS ·  8.1s · 4 ·  6.8k |
| refactor_rename            | PASS · 14.2s · 8 · 16.8k | PASS · 28.4s · 5 · 11.4k |
| search_and_summarize       | PASS ·  5.0s · 4 ·  6.7k | PASS ·  8.9s · 3 ·  5.1k |
| shell_pipeline             | PASS · 17.3s · 8 · 17.2k | PASS · 15.5s · 5 ·  9.4k |
| write_file_from_scratch    | PASS ·  5.8s · 3 ·  5.2k | PASS ·  8.4s · 3 ·  5.2k |
| **Pass rate**              | **9/9**                 | **9/9**                 |

Notes:

- An earlier 4b run the same day failed `shell_pipeline` (max iterations,
  one loop nudge); the other 8 tasks passed. The trap: `write_file` with
  `"10"` leaves no trailing newline, so `cat a.txt b.txt c.txt` yields
  `102012` and `sort -n` sees one line. 4b sometimes recovers, sometimes
  flails. Treat single runs as noisy until `--repeat` lands.
- Both models now saturate the suite; it needs harder tasks to separate
  tiers. 9b uses fewer iterations on multi-step tasks but is ~1.5–5x slower
  per task.
- No JSON repairs or text tool-call recoveries fired in either run.
- The 2026-07 table (4b 5/9, 9b 7/9, 27b 8/9) predates the harness fixes
  (`--model` ignored, a check that could never pass, substring matching)
  and is superseded.
---

## Layout

```
evals/
├── README.md          # this file
├── run.py             # ~200 lines: copy fixture → run agent headless → score
├── tasks/*.toml       # task fixtures — prompt + programmatic check
└── fixtures/          # tiny sample repos used by tasks
```

Each task is a self-contained TOML file (parsed with stdlib `tomllib`):

```toml
id = "refactor_rename"
category = "refactor"
setup = "fixtures/py_project"
timeout_seconds = 240
prompt = """
Rename the function `calc` to `compute` across all .py files in this
workspace and make sure the tests still pass.
"""

[[checks]]
description = "No `def calc` remains"
shell = "! grep -rq 'def calc(' src tests"

[[checks]]
description = "Tests still pass"
shell = "{python} -m pytest -q"
```

Check fields:

| Field | Meaning |
|---|---|
| `shell` | Command run in the workspace; must exit 0. `{python}` is the harness interpreter. Use `'''…'''` literal strings to avoid escaping quotes. |
| `expect_exact` | Stripped stdout must equal this. Prefer it for "write ONLY X" tasks. |
| `expect_regex` | stdout must match this regex. |
| `expect_stdout` / `expect_no_stdout` | stdout must / must not contain this substring. |

Task-level `fixture_unchanged = ["tests"]` fails the task if the agent
modified those fixture paths (caches like `__pycache__` are ignored).

A non-zero exit from `agent-mini chat -m` fails the task with the reason
`max_iterations` (2), `provider_error` (3), `stuck` (4, the agent kept
repeating the same tool calls), `timeout` or `cancelled`.

Each run is isolated from your setup: the agent gets `--yes` (no approval
prompts) and a throwaway `AGENT_MINI_HOME` whose config holds only your
provider settings plus `temperature`, `tier` and `contextWindow`. Your
memory, plugins, sessions and system prompt are not used, and the run
can't write to them.

Add a task: drop a TOML file in `tasks/` and (optionally) a matching
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
| Text tool-call recoveries | How often the model wrote a tool call as text instead of calling it. |
| Loop-detection hits | Are small models stalling? |
| Total tokens / wall time | Cost. |

The per-task `metrics` in the JSON report are scraped from the agent's
output: iterations and tokens from the stats line `chat -m` prints to
stderr (`N iterations  •  X in  •  Y out  •  Ts`), the rest from log lines
(`Repaired malformed tool arguments`, `Recovered N tool call(s)`,
`Loop detected`). A structured trace file is planned.

The **thesis test**: run the same suite across `tiny` / `small` / `medium` /
`cloud` tiers. If tuned `3B` ≈ untuned `8B`, the optimization is proven. If
not, it's theater — and this harness is exactly what surfaces that.

---

## Design constraints

- **No new dependencies.** stdlib only (`subprocess`, `shutil`, `tempfile`,
  `json`, `re`, `argparse`, `tomllib`).
- **No mocking.** Real provider, real tools, real filesystem. Fixtures are
  copied to a fresh `tempfile.mkdtemp()` per task, so runs don't leak.
- **Zero framework bloat.** ~200 LOC in `run.py`. Read it in five minutes.
- **CI-friendly.** Gate as nightly against a live model; keep unit tests as
  the fast job. Model latency should not block PRs.

## Adding a task

1. Create `evals/fixtures/my_fixture/` with a minimal set of files.
2. Create `evals/tasks/my_task.toml`:
   - `id`: matches the file stem.
   - `setup`: path to the fixture (relative to `evals/`).
   - `prompt`: what to ask the agent.
   - `checks`: shell one-liners that must exit `0`.
3. Verify: `python evals/run.py --task my_task`.

Each bug we fix should get a new task. The corpus grows organically.
