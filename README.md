# Agent Mini

[![PyPI version](https://img.shields.io/pypi/v/agent-mini)](https://pypi.org/project/agent-mini/)
[![CI](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A minimal, local-first AI agent you can actually understand and extend.

- **~3,000 lines of Python** — read the whole thing in an afternoon
- **Local-first** — Ollama as default, OpenAI if you want cloud, or any OpenAI-compatible server
- **Zero frameworks** — pure `httpx` + `asyncio`, no LangChain, no LiteLLM
- **Built-in tools** — shell, files, web search, persistent memory
- **Extensible** — drop a Python file in `~/.agent-mini/plugins/` and it's a tool
- **Small-model optimized** — token-aware context pruning, tool call repair, model-tier tuning

## Quick Start

```bash
pip install agent-mini
agent-mini init
agent-mini chat
```

The `init` wizard walks you through picking a provider, model, and basic settings. It creates `~/.agent-mini/config.json` — you're ready to chat.

## Providers

Agent Mini ships with three providers. Set `"provider"` in your config:

### Ollama (default) — Local Models

```bash
ollama pull llama3.1
```

```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434",
      "model": "llama3.1",
      "think": false
    }
  }
}
```

`think` controls thinking mode — `false`, `true`, or `"low"` / `"medium"` / `"high"`.

### OpenAI

```json
{
  "provider": "openai",
  "providers": {
    "openai": {
      "apiKey": "sk-...",
      "model": "gpt-4o"
    }
  }
}
```

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

All providers support **streaming** and **tool calling**.

---

## Tools

Available out of the box — no API keys needed:

| Tool | Description |
|------|-------------|
| `shell_exec` | Run shell commands |
| `read_file` | Read file contents |
| `write_file` | Create / overwrite files |
| `append_file` | Append to files |
| `code_edit` | Find-and-replace in files |
| `list_directory` | Browse filesystem |
| `search_files` | Grep / ripgrep across files |
| `web_search` | DuckDuckGo search (free, no key) |
| `web_fetch` | Fetch any URL as plain text |
| `memory_store` | Save to persistent memory |
| `memory_recall` | Fuzzy search memory (TF-IDF) |

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

---

## Chat Commands

```
/clear              Reset conversation
/model <name>       Switch provider/model (e.g. ollama/llama3.1)
/tools              List available tools
/memory [query]     Browse or search memories
/status             Show config and token usage
/save [file]        Export conversation as Markdown
/sessions           List saved sessions
/load <id>          Resume a session
/help               Show commands
```

Multi-line input: wrap with `"""` or `'''`. Line continuation: end with `\`.

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

---

## Sandbox & Security

Control tool access:

| Level | Description |
|-------|-------------|
| `unrestricted` | All tools, all paths |
| `workspace` | All tools, paths restricted to workspace (default) |
| `readonly` | Read-only — no shell, write, edit |

```json
{ "tools": { "sandboxLevel": "readonly" } }
```

Dangerous shell commands (`rm -rf`, `sudo`, `mkfs`, etc.) are blocked by default.

---

## Sessions

Conversations auto-save after each turn. Resume:

```bash
agent-mini chat -s 20260307_143022
```

Or inside the REPL: `/sessions` to list, `/load <id>` to resume.

---

## How It Works

Agent Mini is a **ReAct loop** — the LLM reasons, picks a tool, observes the result, and repeats until it has an answer.

Key design choices for small/local models:

- **Token-aware context** — estimates token usage and prunes old tool results when approaching the model's effective context window
- **Model tier classification** — auto-detects tiny/small/medium/cloud models and adjusts context budgets, iteration limits, and output caps
- **Tool call repair** — fixes malformed JSON from small models (trailing commas, single quotes, unquoted keys)
- **Loop detection** — catches repeated identical tool calls and nudges the LLM to try a different approach
- **History summarization** — compresses long conversations to stay within context

---

## Configuration Reference

```json
{
  "provider": "ollama",
  "providers": {
    "ollama": { "baseUrl": "http://localhost:11434", "model": "llama3.1", "think": false },
    "openai": { "apiKey": "", "model": "gpt-4o" },
    "local":  { "baseUrl": "http://localhost:8080/v1", "apiKey": "no-key", "model": "local-model" }
  },
  "agent": {
    "maxIterations": 20,
    "temperature": 0.7,
    "systemPrompt": ""
  },
  "channels": {
    "telegram": { "enabled": false, "token": "", "allowFrom": [], "streamResponses": true }
  },
  "tools": { "restrictToWorkspace": false, "sandboxLevel": "workspace", "blockedCommands": [] },
  "memory": { "enabled": true, "maxEntries": 1000 },
  "workspace": "~/.agent-mini/workspace"
}
```

Key paths:
- Config: `~/.agent-mini/config.json`
- Workspace: `~/.agent-mini/workspace/`
- Memory: `~/.agent-mini/memory.json`
- Plugins: `~/.agent-mini/plugins/`
- Sessions: `~/.agent-mini/sessions/`

---

## CLI

| Command | Description |
|---------|-------------|
| `agent-mini init` | Interactive setup wizard |
| `agent-mini chat` | Interactive chat |
| `agent-mini chat -m "..."` | Single message |
| `agent-mini gateway` | Start Telegram bot |
| `agent-mini status` | Show config status |

---

## Project Structure

```
src/agent_mini/
├── cli.py                  # CLI commands (Click)
├── config.py               # Typed config
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
│   ├── openai.py           # OpenAI
│   └── local.py            # OpenAI-compatible
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
