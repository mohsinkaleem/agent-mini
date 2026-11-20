# Agent Mini

[![PyPI version](https://img.shields.io/pypi/v/agent-mini)](https://pypi.org/project/agent-mini/)
[![CI](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinkaleem/agent-mini/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A small AI agent you can read in an afternoon. It runs shell commands, edits files, searches the web and remembers things, using a small model on your own machine or any OpenAI-compatible API.

- **Small:** about 3,100 lines of Python, with only `httpx`, `click` and `rich` as dependencies.
- **Built for small models:** tuned on 4B and 9B models with a [task-eval harness](evals/). On qwen3.5:4b, the pass rate went from 88% to 97% with about a quarter fewer tool calls.
- **Local-first:** uses Ollama by default. OpenAI, LM Studio, vLLM and llama.cpp work too.
- **Guard rails:** a workspace sandbox by default, approval before commands and edits (`agent-mini init` turns it on), and `/undo` for file changes.
- **Extras:** persistent memory, saved sessions, and custom tools as plugins.

## Quick start

```bash
pip install agent-mini
ollama pull qwen3.5:9b      # or qwen3.5:4b on a smaller machine
agent-mini init             # writes ~/.agent-mini/config.json
agent-mini doctor           # checks the provider, model and setup
agent-mini chat
```

Or send one message from a script:

```bash
agent-mini chat -m "How many TODOs are in src/? Reply with just the number." --workspace .
```

## Configuration

Settings live in `~/.agent-mini/config.json`. Everything has a default, so only add what you change. [config.example.json](config.example.json) lists every setting.

```json
{
  "provider": "ollama",
  "providers": { "ollama": { "model": "qwen3.5:9b" } },
  "tools": { "confirm": ["shell", "write_file", "edit_file"] }
}
```

Leave out `tools.confirm` and the agent runs commands and edits files without asking (see [Safety](#safety)).

| Provider | Use it for |
|---|---|
| `ollama` | Ollama (the default) |
| `openai` | OpenAI. The key comes from `apiKey`, `AGENT_MINI_API_KEY` or `OPENAI_API_KEY` |
| `local` | Any OpenAI-compatible server, such as LM Studio, vLLM or llama.cpp. Set `baseUrl` |

To give the agent project instructions, such as build commands or code style, put an `AGENTS.md` file in the workspace.

## Tools

| Tool | What it does |
|---|---|
| `shell` | Runs a command in the workspace |
| `read_file`, `write_file`, `edit_file` | Reads, creates and edits files |
| `list_files`, `search_files` | Finds files by glob and searches their contents |
| `web_search`, `web_fetch` | Searches DuckDuckGo and fetches pages |
| `remember`, `recall` | Saves facts across sessions and searches them |

To add your own tool, put a `.py` file that defines `TOOL_DEF` and `handler` in `~/.agent-mini/plugins/`:

```python
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

In the `readonly` sandbox, a plugin is loaded only if its `TOOL_DEF` has `"x-readonly": true`.

## Chat commands

| Command | What it does |
|---|---|
| `/undo` | Reverts the agent's last file change |
| `/clear` | Starts a new conversation |
| `/model <name>` | Switches the model and saves it to the config |
| `/sessions`, `/load <id>` | Lists saved sessions and resumes one |
| `/help` | Shows all commands |

Ctrl+C stops the current turn. For multi-line input, wrap the text in `"""`.

## Safety

- **Sandbox:** `tools.sandboxLevel` controls what the agent can reach.
  - `workspace` (the default) keeps every path inside the workspace.
  - `readonly` does the same and also removes the shell and file-writing tools.
  - `unrestricted` allows any path.
- **Approval:** tools listed in `tools.confirm` ask before they run. The list is empty unless you fill it; `agent-mini init` puts `shell`, `write_file` and `edit_file` in it. Pass `--yes` to approve everything.
- **Guards:** dangerous commands like `rm -rf` and `sudo` are blocked. The shell doesn't get environment variables that look like secrets, and `web_fetch` refuses private addresses.

The guards reduce risk, but they aren't a security boundary. Approval is the real control.

## Learn how it works

- [docs/LEARN.md](docs/LEARN.md) builds an agent in about 100 lines, then walks through what the real one adds. Start here.
- [docs/how-it-works.html](docs/how-it-works.html) is an interactive tour of how one message flows through the agent.
- [docs/TECHNICAL_IMPLEMENTATION.md](docs/TECHNICAL_IMPLEMENTATION.md) covers the full design, including every small-model fix.
- [evals/README.md](evals/README.md) describes the eval harness and has the latest results.

## Development

```bash
git clone https://github.com/mohsinkaleem/agent-mini.git && cd agent-mini
uv sync --extra dev
uv run pytest
uv run ruff check src tests evals examples
```

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md). Agent Mini is MIT licensed.
