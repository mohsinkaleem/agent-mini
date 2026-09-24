# Agent Mini

[![PyPI version](https://img.shields.io/pypi/v/agent-mini)](https://pypi.org/project/agent-mini/)
[![CI](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A minimal, local-first AI agent you can actually understand and extend.

- **Under 5,000 lines of Python** — read the whole thing in an afternoon
- **Local-first** — Ollama as default, OpenAI if you want cloud, or any OpenAI-compatible server
- **Zero frameworks** — pure `httpx` + `asyncio`, no LangChain, no LiteLLM
- **Built-in tools** — shell, files, web search, persistent memory
- **Vision** — drop an image path or URL into any message; works on Ollama, OpenAI, and OpenAI-compatible providers
- **Extensible** — drop a Python file in `~/.agent-mini/plugins/` and it's a tool
- **Small-model optimized** — tier-scaled system prompt, token-aware context pruning, tool-call repair (including calls written as text), and a [task-eval harness](evals/) to measure whether the tuning actually pays off
- **Safe by default** — sandboxed paths, optional approval before shell commands and file edits, `/undo` for file changes

## Quick Start

```bash
pip install agent-mini
agent-mini init
agent-mini doctor   # checks provider, model, tool support and permissions
agent-mini chat
```

The `init` wizard walks you through picking a provider, model, and basic settings. It creates `~/.agent-mini/config.json` — you're ready to chat.

## Providers

Agent Mini ships with three providers. Set `"provider"` in your config:

### Ollama (default) — Local Models

```bash
ollama pull qwen3:8b
```

```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434",
      "model": "qwen3:8b",
      "think": false
    }
  }
}
```

`think` controls thinking mode — `false`, `true`, or `"low"` / `"medium"` / `"high"`.

Agent Mini asks Ollama for the model's real size, context length and capabilities (`/api/show`), picks the tier from them, and sends a matching `num_ctx` so Ollama doesn't silently cut the prompt. Set `numCtx` to pin the window yourself and `keepAlive` (e.g. `"30m"`) to keep the model loaded. Ollama builds without native tool messages are not supported.

### OpenAI

```json
{
  "provider": "openai",
  "providers": {
    "openai": {
      "apiKey": "sk-...",
      "model": "gpt-5-mini"
    }
  }
}
```

Leave `apiKey` empty to use the `AGENT_MINI_API_KEY` or `OPENAI_API_KEY` environment variable instead.

### Local — Any OpenAI-Compatible Server

Works with LM Studio, vLLM, llama.cpp, text-generation-webui, etc.

```json
{
  "provider": "local",
  "providers": {
    "local": {
      "baseUrl": "http://localhost:8080/v1",
      "apiKey": "no-key",
      "model": "my-model"
    }
  }
}
```

All providers support **streaming** and **tool calling**. Thinking output (`reasoning_content` or Ollama's `thinking`) can be shown with `chat --show-thinking`.

---

## Tools

Available out of the box — no API keys needed:

| Tool | Description |
|------|-------------|
| `shell_exec` | Run shell commands (secret-looking env vars are not passed through) |
| `read_file` | Read a file, optionally a line range (`offset` / `limit`) |
| `write_file` | Create / overwrite files |
| `code_edit` | Find-and-replace in files; tolerates indentation differences and shows the result |
| `list_directory` | Browse filesystem |
| `find_files` | Find files by glob, recursively (`*.py`, `src/**/test_*.py`) |
| `search_files` | Grep / ripgrep across files |
| `web_search` | DuckDuckGo search (free, no key) |
| `web_fetch` | Fetch a public URL as plain text (private/loopback hosts blocked) |
| `memory_store` | Save to persistent memory (same key = update) |
| `memory_recall` | Fuzzy search memory (TF-IDF) |
| `memory_forget` | Delete a memory by key |

### Plugins

Extend with custom tools — drop a `.py` file in `~/.agent-mini/plugins/`:

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

In the `readonly` sandbox, only plugins whose `TOOL_DEF` includes `"x-readonly": true` are loaded.

### Project instructions

Put an `AGENTS.md` (or `.agent-mini.md`, which wins if both exist) in the workspace and it is added to the system prompt: build commands, code style, things to avoid. It is capped at about 1.5k tokens (500 for tiny models).

---

## Chat Commands

```
/clear              Reset conversation
/undo               Revert the last file change made by the agent
/model <name>       Switch provider/model (e.g. ollama/qwen3:8b)
/tools              List available tools
/memory [query]     Browse or search memories
/status             Show config, tier and token usage
/sessions           List saved sessions
/load <id>          Resume a session
/help               Show commands
```

Multi-line input: wrap with `"""` or `'''`. Line continuation: end with `\`. **Ctrl+C** stops the current turn and returns to the prompt; press it at the prompt to quit.

Answers stream as they are generated. `/undo` covers `write_file` and `code_edit` (the last 50 changes of the session), not changes made through `shell_exec`.

---

## Telegram Gateway

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Run `agent-mini init` and enable Telegram during setup, or edit config:

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"],
      "streamResponses": true
    }
  }
}
```

3. `agent-mini gateway`

Use your **numeric** user ID in `allowFrom` (message [@userinfobot](https://t.me/userinfobot) to get it). Usernames work but can be changed and then claimed by someone else. If `allowFrom` is empty (or contains `"*"`), the bot is **public**: the gateway prints a warning, logs every new user, and runs in the `readonly` sandbox unless you set `"allowShell": true`.

The token can also come from the `TELEGRAM_BOT_TOKEN` environment variable. Each user gets a separate conversation, and messages from one user are handled one at a time. When more than one user can reach the bot, memory is turned off so one user's facts don't end up in another's prompt (`"sharedMemory": true` opts back in). Local image paths in Telegram messages are ignored, and tools listed in `tools.confirm` are denied because nobody can approve them.

---

## Sandbox & Security

Control tool access:

| Level | Description |
|-------|-------------|
| `unrestricted` | All tools, all paths |
| `workspace` | All tools, paths restricted to workspace (default) |
| `readonly` | Read-only: no shell, write or edit tools, and paths restricted to workspace |

```json
{ "tools": { "sandboxLevel": "readonly" } }
```

Any other value is a config error: the agent refuses to start rather than running without a sandbox. Blocked tools are not advertised to the model.

### Approval mode

List tools that need your OK before they run:

```json
{ "tools": { "confirm": ["shell_exec", "write_file", "code_edit"] } }
```

The CLI shows the command or file and asks `[y]es / [N]o / [a]lways / [d]iff` (`d` shows a unified diff of the edit). `init` turns this on by default. `chat --yes` approves everything (for scripts); without `--yes`, a non-interactive run denies these tools.

### Other safeguards

Dangerous shell commands (`rm -rf` in any flag order, `sudo`, `mkfs`, etc.) are blocked by default. Treat the shell blocklist as **friction, not a boundary** — a determined caller can bypass it; approval mode is the real control.

`shell_exec` does not pass environment variables whose names contain `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL`; list any the shell needs in `tools.shellEnvAllow`.

`web_fetch` is guarded against basic SSRF: requests to `localhost`, loopback (`::1`), RFC-1918 private ranges, and link-local (including the cloud metadata endpoint `169.254.169.254`) are refused. Redirects are followed one hop at a time and each target is checked **before** it is requested; downloads stop at 2 MB. DNS rebinding can still bypass — use `sandboxLevel: readonly` if you need a stronger guarantee.

Web content reaches the model wrapped in `<untrusted_content>` tags, and the system prompt tells it never to follow instructions found there. This makes prompt injection harder, not impossible.

`config.json`, `memory.json` and session files are written with owner-only (`0600`) permissions.

---

## Sessions

Conversations auto-save after each turn, including one-shot `chat -m` runs. Resume:

```bash
agent-mini chat -s 20260307_143022_a1b2
```

Or inside the REPL: `/sessions` to list, `/load <id>` to resume.

---

## How It Works

Agent Mini is a **ReAct loop** — the LLM reasons, picks a tool, observes the result, and repeats until it has an answer.

Key design choices for small/local models:

- **Tier-scaled system prompt** — tiny models get a compact rules block and no memory recall; larger tiers get the full descriptive prompt and recent context
- **Inline tool list** — `<available_tools>` block in the system prompt so small models can see tool names at a glance without inferring from the JSON schema
- **Token-aware context** — estimates token usage and prunes old tool results when approaching the model's effective context window
- **Model tier classification** — reads the parameter count from the model name (`:8b`, `:0.6b`, `8x7b`) and sorts models into tiny (<4B) / small (4–8B) / medium (9–19B) / large (20–72B) / cloud (bigger, or API models such as `gpt-5`, `o3`, `claude-*`). With Ollama the size reported by the server wins over the name. The tier sets context budgets, iteration limits, and output caps. Override with `agent.tier` and `agent.contextWindow`
- **Stable prompt prefix** — the system prompt changes only when the date does, and volatile parts (memories) come last, so local servers can reuse their KV cache between turns
- **Tool call repair** — fixes malformed JSON from small models (trailing commas, single quotes, unquoted keys), and recovers tool calls a model wrote as text (`<tool_call>` tags, `[TOOL_CALLS]`, fenced or bare JSON)
- **Budget-based pruning** — before each call, older tool results are trimmed and, if the request is still over the tier's budget, replaced with a one-line stub
- **Loop detection** — catches repeated calls (same call and result, or A→B→A→B), nudges once, then stops and asks the model to explain
- **Graceful stops** — on hitting the iteration limit the model gets one last call without tools to summarize what it did and what's left
- **Tool trace** — each saved reply notes which tools ran (`[tools used: read_file(a.py), …]`), so follow-up turns know what was touched
- **History summarization** — compresses long conversations to stay within context
- **[Task-eval harness](evals/)** — measures success rate, iterations, tokens, and JSON-repair fire-rate across model tiers so the tuning above is testable, not hand-waved

---

## Configuration Reference

```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434", "model": "qwen3:8b", "think": false,
      "numCtx": null, "keepAlive": null
    },
    "openai": { "apiKey": "", "model": "gpt-5-mini", "reasoningEffort": null },
    "local":  { "baseUrl": "http://localhost:8080/v1", "apiKey": "no-key", "model": "local-model" }
  },
  "agent": {
    "maxIterations": null,
    "temperature": 0.7,
    "systemPrompt": "",
    "tier": null,
    "contextWindow": null,
    "showThinking": false
  },
  "channels": {
    "telegram": {
      "enabled": false, "token": "", "allowFrom": [], "allowShell": false,
      "sharedMemory": false, "streamResponses": true
    }
  },
  "tools": {
    "sandboxLevel": "workspace",
    "confirm": ["shell_exec", "write_file", "code_edit"],
    "shellTimeout": 120,
    "shellEnvAllow": [],
    "blockedCommands": []
  },
  "memory": { "enabled": true, "maxEntries": 1000 },
  "workspace": "~/.agent-mini/workspace"
}
```

Key paths (all under `~/.agent-mini/`, or `$AGENT_MINI_HOME` if set):
- Config: `config.json`
- Workspace: `workspace/`
- Memory: `memory.json`
- Plugins: `plugins/`
- Sessions: `sessions/`

Notes:
- `agent.temperature: null` sends no temperature. OpenAI reasoning models (`o*`, `gpt-5*`) never get one, since they reject it.
- `agent.tier` forces a tier (`tiny`, `small`, `medium`, `large`, `cloud`); `agent.contextWindow` overrides the tier's context budget. `agent.maxIterations: null` uses the tier's limit (10–25).
- `memory.enabled: false` removes the memory tools and keeps stored memories out of the prompt.
- `tools.confirm: []` (the default for configs written before approval mode existed) runs every tool without asking.
- Secrets can come from the environment instead of the file: `AGENT_MINI_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`.
- `restrictToWorkspace` is deprecated; use `sandboxLevel`.

---

## CLI

| Command | Description |
|---------|-------------|
| `agent-mini init` | Interactive setup wizard |
| `agent-mini doctor` | Check config, provider, model, tool support and permissions (exit `1` if something is broken) |
| `agent-mini chat` | Interactive chat |
| `agent-mini chat -m "..."` | Single message (`-m -` reads stdin). Exit code: `0` ok, `2` max iterations, `3` provider error, `4` stuck in a loop, `130` interrupted |
| `agent-mini chat --provider <p> --model <m>` | Override provider / model for one run (not saved) |
| `agent-mini chat --workspace <dir>` | Override the workspace for a single run (also honours `AGENT_MINI_WORKSPACE`) |
| `agent-mini chat --yes` | Approve every tool call without asking |
| `agent-mini chat --no-stream` / `--show-thinking` | Print the answer only when complete / stream the model's thinking |
| `agent-mini gateway` | Start Telegram bot |

---

## Project Structure

```
src/agent_mini/
├── cli.py                  # CLI commands (Click)
├── config.py               # Config loading
├── bus.py                  # Message routing
├── sessions.py             # Session persistence
├── agent/
│   ├── loop.py             # ReAct agent loop
│   ├── context.py          # System prompt builder
│   ├── memory.py           # JSON memory + TF-IDF search
│   ├── tools.py            # Built-in tools + plugin loader
│   ├── token_estimator.py  # Token counting + model tiers
│   └── vision.py           # Image detection + encoding
├── providers/
│   ├── base.py             # Provider interface
│   ├── ollama.py           # Ollama
│   └── local.py            # OpenAI and OpenAI-compatible
└── channels/
    ├── base.py             # Channel interface
    └── telegram.py         # Telegram bot
```

## Development

```bash
git clone https://github.com/mohsinkaleem/agent-mini.git
cd agent-mini
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

### Task evals

The [`evals/`](evals/) directory contains a small, framework-free task-eval
harness that runs the agent end-to-end against real fixtures (refactor a
codebase, fix a failing test, answer a codebase question, etc.). Use it to
measure whether the small-model optimizations actually pay off:

```bash
python evals/run.py                        # run all tasks against config model
python evals/run.py --task refactor_rename # single task
python evals/run.py --compare results/*.json  # cross-tier comparison table
```

See [evals/README.md](evals/README.md) for the full workflow.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
