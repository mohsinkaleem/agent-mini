# Agent Mini — Task Evals

Unit tests check the code; these check the **agent**: the real CLI, a real model, real files. Each task copies a small fixture into a temp workspace, runs `agent-mini chat -m` headless, then scores what the agent left behind and what it answered. Stdlib only.

```bash
python evals/run.py                                   # every task once, with your configured model
python evals/run.py --model qwen3.5:4b --repeat 3     # three runs per task (small models are noisy)
python evals/run.py --category simple                 # chat | simple | multi_step | robustness | safety
python evals/run.py --task log_tail,exit_code
python evals/run.py --set agent.temperature=0.2       # A/B any config value
python evals/run.py --compare results/a.json results/b.json
python evals/run.py --validate                        # check the checks (no model needed; pytest runs this)
```

Use the project's environment (`uv run python evals/run.py …`) so `{python}` in checks has pytest.

## What a run does

1. Copies `fixtures/<setup>` into a fresh temp workspace and runs `setup_shell` there.
2. Writes a throwaway `AGENT_MINI_HOME` with **only your provider settings** (provider, model, URL, key) plus the task's `config` and your `--set` overrides. Everything else is the code's defaults, so your memory, plugins, sessions and system prompt neither affect the run nor get written to.
3. Runs `python -m agent_mini chat -m <prompt> --yes --report <file>` (and each `followups` prompt in the same session, or a new one with `new_session = true`). On timeout it sends Ctrl+C first, so the report still holds the transcript.
4. Scores the run. It fails on the first of: a non-zero exit (`max_iterations`, `provider_error`, `stuck`, `timeout`), a modified `fixture_unchanged` path, a `forbidden_tools` call, more than `max_tool_calls`, or a failing check.

Failed runs keep their workspace and write the whole transcript to `results/failures/<label>/<task>-<time>.json`. Read those first: every small-model fix in 0.5 came from one.

## Task format

```toml
id = "edit_setting"
category = "simple"
timeout_seconds = 90
max_tool_calls = 4                 # a simple job done in many calls is a failure
setup_shell = "printf '[server]\\ndebug = true\\n' > settings.ini"
solution = "printf '[server]\\ndebug = false\\n' > settings.ini"   # for --validate
prompt = "In settings.ini, change debug from true to false. Do not change anything else."

[[checks]]
description = "debug is false and nothing else changed"
shell = "cat settings.ini"
expect_exact = "[server]\ndebug = false"
```

| Task field | |
|---|---|
| `setup` / `setup_shell` | fixture directory to copy / shell to run in the workspace first |
| `followups`, `new_session` | more prompts after the first; `true` gives each one a fresh session |
| `config` | merged into the agent config, e.g. `{ memory = { enabled = true } }` |
| `max_tool_calls`, `forbidden_tools` | efficiency and safety limits, over all turns |
| `fixture_unchanged` | paths the agent must not modify (caches ignored) |
| `solution`, `solution_answer` | a reference solution: shell that solves it, and/or a correct reply |

A check with `shell` runs it in the workspace (it must exit 0) and matches its stdout; a check **without** `shell` matches the agent's final reply. Matchers: `expect_exact` (after strip), `expect_regex`, `expect_stdout` (substring), `expect_no_stdout`. `{python}` is the harness's interpreter. Use `'''…'''` TOML strings to avoid escaping.

`--validate` runs every task's checks twice without a model: on the untouched fixture they must **fail**, and after `solution` (with `solution_answer` as the reply) they must **pass**. That catches a check that can never pass, or one that passes no matter what.

## The tasks

| Category | Tasks | What they catch |
|---|---|---|
| `chat` (4) | capital, greeting, arithmetic, explain a command | answering without tools when none are needed |
| `simple` (18) | read a fact, count rows, read a JSON field, find a definition, create/rename/sort/append/edit a file, write JSON, count files, day of week, … | doing one-step jobs in one or two calls, with the right paths |
| `multi_step` (8) | fix a failing test, fix a bug in another file, implement from tests, rename across files, shell pipeline, follow-up turn, memory across sessions, find a symbol in 60 files | planning, verifying, staying within budget |
| `robustness` (4) | tail of a 3,000-line log, byte-exact quotes/backslashes/unicode, tab-indented Makefile, a command that fails silently | truncation, JSON escaping, whitespace, exit codes |
| `safety` (2) | prompt injection in a README, a write request in the readonly sandbox | not following file contents; saying "I can't" instead of pretending |

## Metrics

Per run (in the JSON report): pass/fail and the reason, seconds, tool calls and their names, tool errors, iterations, tool calls recovered from text, loop nudges, prompt and completion tokens, finish reason. Per report: pass rate, tasks passed in every run, pass rate per category, mean seconds, tool calls and prompt tokens. `--compare` puts several reports side by side, per task (`passed/runs` and mean calls) and per category.

## Results (2026-10-14)

Ollama 0.40 on an M4 Pro (48 GB), `think: false`, tier defaults (temperature 0.7), 36 tasks × 3 runs per model.
**baseline** = v0.4.0 (with only `--report` added); **cand1** = the 0.5 rewrite; **cand2** = cand1 plus the
review fixes, the read-before-overwrite guard, tab-safe fuzzy edits and negative `read_file` offsets.

| | 4b baseline | 4b cand1 | 4b cand2 | 9b baseline | 9b cand1 |
|---|---|---|---|---|---|
| Pass rate (runs) | 88.0% (95/108) | 96.3% (104/108) | **97.2%** (105/108) | 97.2% (105/108) | 96.3% (104/108) |
| Tasks passed in all 3 runs | 27/36 | 32/36 | **34/36** | 34/36 | 34/36 |
| chat | 12/12 | 12/12 | 12/12 | 12/12 | 9/12 |
| simple | 44/54 | 53/54 | 52/54 | 51/54 | **54/54** |
| multi_step | 23/24 | 23/24 | 23/24 | 24/24 | 23/24 |
| robustness | 11/12 | 10/12 | **12/12** | 12/12 | 12/12 |
| safety | 5/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Tool calls per task | 3.0 | 2.4 | **2.3** | 2.7 | **2.2** |
| Seconds per task | 10.3 | 5.5 | 5.8 | 11.4 | **9.5** |
| Prompt tokens per task | 6,608 | 5,339 | **5,121** | 6,275 | **4,831** |
| Timeouts | 3 | 0 | 0 | 0 | 0 |

What the transcripts showed and what changed because of it:

- **Root-relative paths** (`/config.json`, `/`, `/workspace`) were the biggest single cause of 4b failures:
  each "Access denied" cost 3–4 calls. `write_json` went 0/3 → 3/3, `json_version` 2/3 → 3/3,
  `edit_setting` 1/3 → 3/3.
- **"Rename" by copying** (read + write, original left behind): the prompt now says to use the shell for moving
  and deleting files. `rename_file` 1/3 → 3/3 on both models.
- **`**/` did not match zero folders** (a bug in `find_files`, kept in the rewrite until the harness caught it):
  `count_py_files` 2/3 → 3/3.
- **Blind overwrites**: 9b "appended" to a script by writing only the new line. `write_file` now refuses to
  replace a file it hasn't read.
- **Tab-stripping fuzzy edits** and **negative offsets** (`tabs_edit`, `log_tail`): fixed in cand2; 4b robustness 12/12.
- **An "answer directly when you can" rule** (cand1/cand2) made 9b answer `17 * 23` from memory, wrongly (0/3), and
  4b guess weekdays; without it, the baseline already answered plain questions with no tools. It was removed. On
  the 26 tasks cand2 finished on 9b before the run was stopped, it passed every run except `chat_math` (0/3) and
  `day_of_week` (2/3), both caused by that rule.

**Not measured yet:** the final prompt wording (the rule above removed, `System:` line and `python3 -c` hint
added), 27B models, and temperatures other than 0.7. To check the prompt change quickly:

```bash
uv run python evals/run.py --model qwen3.5:9b --repeat 3 --task chat_capital,chat_explain,chat_greeting,chat_math,day_of_week
```

The 4b failures left in cand2 are `day_of_week` (BSD `date` has no `-d`, the reason for the `System:` line) and
`shell_pipeline` (files written without a trailing newline make `cat` print `102012`; the model notices but
fumbles the fix).
