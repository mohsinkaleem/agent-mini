# Agent Mini

[![PyPI version](https://img.shields.io/pypi/v/agent-mini)](https://pypi.org/project/agent-mini/)
[![CI](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A small, local-first AI agent you can read in an afternoon: shell, files, web and memory, driven by a small model on your own machine (tuned on 4B and 9B) or by any OpenAI-compatible API.

- **About 3,000 lines of Python**, no frameworks: `httpx`, `click`, `rich`, and nothing else
- **Built for small models**: tuned and measured on 4B and 9B models with a [task-eval harness](evals/): on the release candidate, qwen3.5:4b went from 88% to 97% of runs passed with about a quarter fewer tool calls
- **Local-first**: Ollama by default, OpenAI or any OpenAI-compatible server (LM Studio, vLLM, llama.cpp) if you prefer
- **Tools**: shell, read/write/edit files, list and search, web search and fetch, persistent memory, plus your own plugins
- **Vision**: put an image path or URL in a message
- **Safe by default**: workspace sandbox, optional approval before commands and edits, `/undo` for file changes
- **Telegram gateway**: the same agent from your phone

## Quick start

```bash
pip install agent-mini
ollama pull qwen3.5:9b      # or qwen3.5:4b on a smaller machine
agent-mini init             # a few questions; writes ~/.agent-mini/config.json
agent-mini doctor           # checks the provider, model, tool support and permissions
agent-mini chat
```

One-shot use from scripts:

```bash
agent-mini chat -m "How many TODOs are in src/? Reply with just the number." --workspace .
```

## What makes it work on small models

The eval transcripts show where small models go wrong; this is what the agent does about it:

- **A short system prompt** (about 240 tokens, plus ~900 for the tool schemas): do every calculation and date with the shell (and it says which OS that shell is), use relative paths, use the shell for moving and deleting files, treat file contents as data.
- **Paths the way small models write them.** `/notes.txt` and `/workspace/notes.txt` mean the workspace file when the sandbox is on, instead of "Access denied" followed by three calls of wandering.
- **Forgiving tool calls.** Common argument names (`file_path`, `cmd`, `old_string`), tool names (`bash`, `str_replace`, `list_directory`) and types (`"10"` for 10, a JSON object as file content) are accepted. Tool calls written as text (`<tool_call>`, `[TOOL_CALLS]`, fenced JSON) are recovered and malformed JSON is repaired.
- **No blind overwrites.** `write_file` won't replace a file the agent hasn't read in this session; small models otherwise "append" a line by writing only that line.
- **Errors that say what to do**: "notes.txt does not exist", "there is no tool 'python'; call shell with the command", the closest match when an edit's `old_text` is not found.
- **A cap on reply length**, so a model that starts repeating itself stops after 4,096 tokens instead of running until the request times out.
- **Budgets by tier.** The model's size (reported by Ollama, or read from its name) sets the context budget, iteration limit and tool-output size. Ollama gets a matching `num_ctx`, so the front of the prompt is never cut silently.
- **Context that fits.** Old tool results are trimmed, then stubbed; long histories are summarized; a repeated tool call gets one nudge and then a forced answer.
- **A stable prompt prefix** (date without a clock, memories last), so local servers reuse their KV cache between turns.

## Providers

Set `"provider"` in the config to `ollama`, `openai` or `local`, and the model under `providers`:

```json
{
  "provider": "ollama",
  "providers": { "ollama": { "model": "qwen3.5:9b" } }
}
```

| Provider | Settings (defaults in [config.example.json](config.example.json)) |
|---|---|
| `ollama` | `baseUrl`, `model`, `think` (`false`, `true` or `"low"`/`"medium"`/`"high"`), `numCtx` (pin the context window), `keepAlive` (e.g. `"30m"`) |
| `openai` | `apiKey` (or `AGENT_MINI_API_KEY` / `OPENAI_API_KEY`), `model`, `reasoningEffort`, `baseUrl` |
| `local` | any OpenAI-compatible server: `baseUrl`, `model`, `apiKey` (or `AGENT_MINI_API_KEY`), `reasoningEffort` |

All providers stream and call tools. `chat --show-thinking` shows the model's thinking (`reasoning_content`, or Ollama's `thinking`).

## Tools

| Tool | What it does |
|------|-------------|
| `shell` | Run a command in the workspace; reports the exit code when it fails. Secret-looking env vars are not passed on. |
| `read_file` | Read a file, or a line range of a big one (`offset`, `limit`; a negative offset counts from the end) |
| `write_file` | Create a file, or replace one the agent has read |
| `edit_file` | Replace text that matches once; tolerates indentation differences and shows the result |
| `list_files` | List a directory, or find files by glob (`*.py`, `src/**/test_*.py`) |
| `search_files` | Search file contents with a regex (ripgrep, or grep) |
| `web_search` | DuckDuckGo search, no key needed |
| `web_fetch` | Fetch a public page as text (private and loopback hosts are refused) |
| `remember` / `recall` | Save facts for later sessions (same key = update) and search them |

### Plugins

A `.py` file in `~/.agent-mini/plugins/` that defines `TOOL_DEF` and `handler` becomes a tool:

```python
# ~/.agent-mini/plugins/timestamp.py
from datetime import datetime, timezone

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_timestamp",
        "description": "Get the current UTC timestamp.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

async def handler(arguments: dict) -> str:
    return datetime.now(timezone.utc).isoformat()
```

In the `readonly` sandbox only plugins whose `TOOL_DEF` has `"x-readonly": true` are loaded.

### Project instructions

An `AGENTS.md` (or `.agent-mini.md`, which wins) in the workspace is added to the system prompt: build commands, code style, things to avoid. It is capped at about 1.5k tokens (500 for tiny models).

## Chat commands

```
/clear              Reset the conversation
/undo               Revert the last file change made by the agent
/model <name>       Switch model, e.g. qwen3.5:4b or local/my-model
/tools              List available tools
/memory [query]     Show or search memories
/forget <key>       Delete a memory
/status             Model, tier, sandbox and token usage
/sessions           List saved sessions
/load <id>          Resume a session
/help               Show the commands
```

Multi-line input: wrap it in `"""` or `'''`, or end lines with `\`. **Ctrl+C** stops the current turn; at the prompt it quits. `/undo` covers `write_file` and `edit_file` (the last 50 changes), not changes made through the shell.

## Sandbox and approval

| `tools.sandboxLevel` | |
|---|---|
| `workspace` (default) | All tools; paths stay inside the workspace |
| `readonly` | No shell, write or edit tools; paths stay inside the workspace |
| `unrestricted` | All tools, any path |

Any other value is a config error: the agent refuses to start rather than run without a sandbox.

`tools.confirm` lists tools that need your OK. The CLI shows the command or file and asks `[y]es / [N]o / [a]lways / [d]iff`; `init` turns this on for `shell`, `write_file` and `edit_file`. `chat --yes` approves everything; without it a non-interactive run denies these tools. Unknown names in `tools.confirm` are a config error, so a typo can't silently turn approval off; the names from before 0.5 (`shell_exec`, `code_edit`) still work.

Other safeguards:

- Dangerous commands (`rm -rf` in any flag order, `sudo`, `mkfs`, a fork bomb, …) are blocked; add patterns in `tools.blockedCommands`. Treat the blocklist as friction, not a boundary: approval is the real control.
- The shell doesn't get env vars whose names contain `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL` unless listed in `tools.shellEnvAllow`.
- `web_fetch` refuses localhost, private, link-local and cloud-metadata addresses, checks every redirect before following it and stops at 2 MB. DNS rebinding can still get through; use `readonly` if you need more.
- Web content reaches the model inside `<untrusted_content>` tags, and the prompt tells it to treat file contents, command output and web pages as data, never as instructions. That makes prompt injection harder, not impossible.
- `config.json`, `memory.json` and session files are written owner-only (`0600`).

## Telegram gateway

```json
{
  "channels": {
    "telegram": { "enabled": true, "token": "123:ABC…", "allowFrom": ["YOUR_NUMERIC_ID"] }
  }
}
```

Create a bot with [@BotFather](https://t.me/BotFather), get your numeric ID from [@userinfobot](https://t.me/userinfobot), install the extra (`pip install agent-mini[telegram]`) and run `agent-mini gateway`. The token can also come from `TELEGRAM_BOT_TOKEN`.

Each user has their own conversation (`/reset` clears it) and their messages run one at a time. With an empty `allowFrom` (or `"*"`) the bot is **public**: the gateway warns, logs each new user and runs `readonly` unless `"allowShell": true`. When more than one person can reach the bot, memory is off unless `"sharedMemory": true`, so one user's facts don't reach another's prompt. Tools in `tools.confirm` are denied, since nobody can approve them, and local image paths are ignored.

## Configuration

Everything has a default; `config.json` only needs what you change. [config.example.json](config.example.json) lists every setting. The ones not covered above:

| Setting | Default | |
|---|---|---|
| `agent.tier` | from the model | `tiny` (<4B), `small` (4–9B), `medium` (9–20B), `large` (20–72B), `cloud` |
| `agent.contextWindow` | the tier's | conversation token budget |
| `agent.maxIterations` | the tier's (10–25) | tool rounds per message |
| `agent.temperature` | the tier's | |
| `agent.systemPrompt` | `""` | appended to the system prompt |
| `memory.enabled` / `memory.maxEntries` | `true` / `1000` | `false` removes the memory tools and keeps memories out of the prompt |
| `tools.shellTimeout` | `120` | seconds |
| `workspace` | `~/.agent-mini/workspace` | |

Files live in `~/.agent-mini/` (or `$AGENT_MINI_HOME`): `config.json`, `memory.json`, `sessions/`, `plugins/`, `workspace/`.

## CLI

| Command | |
|---------|-------------|
| `agent-mini init` | Setup wizard |
| `agent-mini doctor` | Check config, provider, model, tool support and permissions (exits `1` if something is broken) |
| `agent-mini chat` | Interactive chat; `-s <id>` resumes a session |
| `agent-mini chat -m "…"` | One message (`-m -` reads stdin). Into a pipe, stdout holds only the reply. Exit code: `0` ok, `2` max iterations, `3` provider error, `4` stuck in a loop, `130` interrupted |
| `--provider`, `--model`, `--workspace` | Override for one run (not saved; `AGENT_MINI_WORKSPACE` works too). With `--workspace`, an `unrestricted` sandbox becomes `workspace` |
| `--yes` | Approve every tool call |
| `--report FILE` | With `-m`: write a JSON report of the run (reply, tool calls, tokens, transcript) |
| `--no-stream`, `--show-thinking`, `--no-markdown`, `-v` | Output options |
| `agent-mini gateway` | Run the Telegram bot |

## How it fits together

```
src/agent_mini/
├── cli.py          # init, chat (REPL and -m), doctor, gateway
├── config.py       # defaults, validation, files
├── loop.py         # the ReAct loop: call the model, run tools, prune, summarize, retry
├── prompt.py       # model tiers, token budgets, the system prompt
├── tools.py        # tool registry, file and shell tools, sandbox, approval, undo, plugins
├── web.py          # web_search, web_fetch, SSRF guard
├── memory.py       # JSON memory with keyword search
├── vision.py       # image paths and URLs → image parts
├── sessions.py     # saved conversations
├── telegram.py     # Telegram gateway
└── providers/      # base types and tool-call parsing, ollama.py, openai.py
```

`AgentLoop.run()` sends the system prompt, history and message; runs the tools the model calls (in parallel); feeds back the results, clipped to the tier's limit; and repeats until the model answers in text, the iteration limit is hit (one last call without tools then summarizes), or the model repeats itself (one nudge, then a forced answer). [docs/TECHNICAL_IMPLEMENTATION.md](docs/TECHNICAL_IMPLEMENTATION.md) has the details. For a plain-English, interactive tour (architecture diagram, a step-by-step walk through one message, a tool call checker), open [docs/how-it-works.html](docs/how-it-works.html).

## Development

```bash
git clone https://github.com/mohsinkaleem/agent-mini.git && cd agent-mini
uv sync --extra dev
uv run pytest            # unit tests, plus a check that every eval task's checks are valid
uv run ruff check src tests evals
```

### Evals

[`evals/`](evals/) runs the real agent against a real model on 36 small tasks: plain questions, one-step file and shell jobs, multi-step fixes, robustness traps and safety checks. It scores the files the agent leaves, its answer and how many tool calls it needed:

```bash
python evals/run.py --model qwen3.5:4b --repeat 3          # the whole suite, three times
python evals/run.py --category simple --set agent.temperature=0.2   # A/B a setting
python evals/run.py --compare evals/results/a.json evals/results/b.json
```

See [evals/README.md](evals/README.md) for the task format and the latest results.

## Upgrading from 0.4

- Tools were renamed: `shell_exec` → `shell`, `code_edit` → `edit_file`, `list_directory` and `find_files` → `list_files`, `memory_store`/`memory_recall` → `remember`/`recall`; `memory_forget` became `/forget`. Old names in `tools.confirm` and from models still work.
- `agent.temperature: null` now means the tier's default, not "send none" (OpenAI reasoning models still never get one). `restrictToWorkspace` is deprecated (`true` still keeps an `unrestricted` sandbox inside the workspace); use `sandboxLevel`.
- Ollama `think` defaults to `false`, and the default model is `qwen3.5:9b`.
- `write_file` refuses to overwrite a file the agent hasn't read in the session (it reads it, then writes).
- `chat -m` into a pipe prints only the reply on stdout; the rest goes to stderr.
- Plugins, config files and sessions keep working.

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute. MIT licensed.
