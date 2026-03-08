# Agent Mini

**Ultra-lightweight personal AI agent** — inspired by [nanobot](https://github.com/HKUDS/nanobot), built lean.

- ~2,000 lines of Python (core agent)
- 4 LLM providers: **Ollama**, **Gemini**, **GitHub Copilot**, **Local** (any OpenAI-compatible)
- 1 chat channel: **Telegram**
- Built-in tools: shell, files, web search & browse, persistent memory
- Zero-framework: pure `httpx` + `asyncio` — no LangChain, no LiteLLM
- **Free web search** — DuckDuckGo scraping, no API key required
- **Streaming** — real-time token output from all providers
- **Session persistence** — resume conversations across restarts
- **Plugin system** — extend with custom tools
- **Vision support** — send images to multi-modal models
- **Token tracking** — per-turn and per-session usage stats

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-repo/agent-mini.git
cd agent-mini

# Option A: Install and link for development (changes in src/ take effect immediately)
uv tool install --editable ".[all]"

# Option B: Standard global install (copies files, no linking)
uv tool install ".[all]"

# Or for local development only:
uv sync --extra all
```

### 2. Initialise

```bash
agent-mini init
```

This creates `~/.agent-mini/config.json` — edit it to set your provider and keys.

### 3. Chat

```bash
# Interactive mode
agent-mini chat

# Single message
agent-mini chat -m "What's the weather in London?"

# Resume a previous session
agent-mini chat -s 20260307_143022
```

### 4. Gateway (Telegram)

```bash
agent-mini gateway
```

---

## Providers

Set `"provider"` in config to one of these, then configure its section under `"providers"`:

| Provider | Description | Streaming | Vision |
|----------|-------------|-----------|--------|
| **Ollama** | Local models via Ollama | ✅ | — |
| **Gemini** | Google Generative AI | ✅ | ✅ |
| **GitHub Copilot** | Copilot chat API (OAuth) | ✅ | ✅ |
| **Local** | Any OpenAI-compatible endpoint | ✅ | ✅ |

All providers support streaming responses and tool calling.

### Ollama (default)

```bash
# Install & run Ollama: https://ollama.ai
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

`think` controls Ollama thinking mode:
- `false` (default): thinking off
- `true`: thinking on
- `"low" | "medium" | "high"`: GPT-OSS thinking levels

References: [Ollama thinking docs](https://docs.ollama.com/capabilities/thinking), [Ollama blog post](https://ollama.com/blog/thinking)

### Gemini

Get an API key at [aistudio.google.com](https://aistudio.google.com/).

```json
{
  "provider": "gemini",
  "providers": {
    "gemini": {
      "apiKey": "AIza...",
      "model": "gemini-2.0-flash"
    }
  }
}
```

### GitHub Copilot

Requires a GitHub account with Copilot access.

```bash
# Interactive OAuth login
agent-mini login github_copilot
```

```json
{
  "provider": "github_copilot",
  "providers": {
    "github_copilot": {
      "model": "gpt-4o"
    }
  }
}
```

### Local (LM Studio / vLLM / llama.cpp)

Any server that implements the OpenAI chat completions API.

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

---

## Chat Channels

### Telegram

1. Create a bot via [@BotFather](https://t.me/BotFather) → copy the token
2. Get your User ID (Settings → or send a message to [@userinfobot](https://t.me/userinfobot))
3. Configure:

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

4. Run: `agent-mini gateway`

`streamResponses: true` enables real-time streamed Telegram replies (works with all providers).

---

## Tools

The agent has these built-in tools (all available out of the box — no API keys needed):

| Tool | Description |
|------|-------------|
| `shell_exec` | Run any shell command |
| `read_file` | Read file contents |
| `append_file` | Append content to a file |
| `write_file` | Create/overwrite files |
| `code_edit` | Targeted find-and-replace (safer than write_file) |
| `list_directory` | Browse the filesystem |
| `search_files` | Search text/regex across files (uses `rg` or `grep`) |
| `web_search` | Search the web via DuckDuckGo (free, no API key) |
| `web_fetch` | Fetch & read any web page as plain text |
| `memory_store` | Save info to persistent memory |
| `memory_recall` | Fuzzy search persistent memory (TF-IDF) |

### Web Search & Browsing

Web search uses **DuckDuckGo HTML scraping** — free with no API key needed. It works out of the box.

The agent can:
1. **Search** — `web_search("Python asyncio tutorial")` returns titles, URLs, and snippets
2. **Read** — `web_fetch("https://example.com/article")` fetches a page and extracts readable text

### Vision

For providers supporting images (Gemini, Copilot, Local with vision models), include image paths or URLs in your message:

```
Describe what you see in /path/to/screenshot.png
What's in this image? https://example.com/photo.jpg
```

Images are automatically detected, encoded, and sent to the model's multi-modal API.

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/clear` | Reset conversation |
| `/model <name>` | Switch provider/model (e.g. `gemini/gemini-2.0-flash`) |
| `/tools` | List available tools with descriptions |
| `/memory [query]` | Browse or search stored memories |
| `/status` | Show current config, token usage |
| `/save [file]` | Export conversation as Markdown |
| `/sessions` | List saved sessions |
| `/load <id>` | Resume a saved session |
| `/help` | Show all commands |

### Multi-Line Input

Start with `"""` or `'''` and end with the same delimiter:

```
"""
def hello():
    print("world")
"""
```

Or use `\` for line continuation.

---

## Sandbox Levels

Control tool access with `tools.sandboxLevel` in config:

| Level | Description |
|-------|-------------|
| `unrestricted` | All tools, all paths |
| `workspace` | All tools, paths restricted to workspace (default) |
| `readonly` | Read-only — no shell, write_file, append_file, or code_edit |

```json
{
  "tools": {
    "sandboxLevel": "readonly"
  }
}
```

### Command Blocklist

Dangerous shell commands are blocked by default (`rm -rf`, `sudo`, `mkfs`, etc.). Add custom patterns:

```json
{
  "tools": {
    "blockedCommands": ["\\bcurl\\b", "\\bwget\\b"]
  }
}
```

---

## Sessions

Conversations are automatically saved after each turn. Resume with:

```bash
# List saved sessions
agent-mini chat
/sessions

# Resume by ID
agent-mini chat -s 20260307_143022

# Or inside the REPL
/load 20260307_143022
```

Sessions are stored in `~/.agent-mini/sessions/`.

---

## Plugins

Extend the agent with custom tools by placing Python files in `~/.agent-mini/plugins/`.

Each plugin file must export:
- `TOOL_DEF` — an OpenAI function-calling tool definition dict
- `handler` — an async (or sync) function that takes `arguments: dict` and returns a string

Example plugin (`~/.agent-mini/plugins/timestamp.py`):

```python
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_timestamp",
        "description": "Get the current UTC timestamp.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

async def handler(arguments: dict) -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
```

Plugins are discovered at startup and available alongside built-in tools.

---

## Token Tracking

Token usage is displayed after each response (when reported by the provider):

```
  tokens: 1250→ 380← (1630 total)
```

Use `/status` to see cumulative session totals.

---

## Configuration Reference

Full config with all options:

```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434",
      "model": "llama3.1",
      "think": false
    },
    "gemini": {
      "apiKey": "",
      "model": "gemini-2.0-flash"
    },
    "github_copilot": {
      "token": "",
      "model": "gpt-4o"
    },
    "local": {
      "baseUrl": "http://localhost:8080/v1",
      "apiKey": "no-key",
      "model": "local-model"
    }
  },
  "agent": {
    "maxIterations": 20,
    "temperature": 0.7,
    "systemPrompt": ""
  },
  "channels": {
    "telegram": {
      "enabled": false,
      "token": "",
      "allowFrom": [],
      "streamResponses": true
    }
  },
  "tools": {
    "restrictToWorkspace": false,
    "sandboxLevel": "workspace",
    "blockedCommands": []
  },
  "memory": {
    "enabled": true,
    "maxEntries": 1000
  },
  "workspace": "~/.agent-mini/workspace"
}
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_phases.py -v
```

This is a lightweight pure-`httpx` approach: no browser process, no Playwright/Selenium, no headless Chrome. HTML is parsed via Python's built-in `html.parser` and converted to clean readable text.

### Security

Set `"restrictToWorkspace": true` to sandbox file and shell operations to the workspace directory:

```json
{
  "tools": {
    "restrictToWorkspace": true
  }
}
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `agent-mini init` | Create config and workspace |
| `agent-mini chat` | Interactive chat |
| `agent-mini chat -m "..."` | Single message |
| `agent-mini gateway` | Start Telegram gateway |
| `agent-mini login github_copilot` | OAuth login for GitHub Copilot |
| `agent-mini status` | Show config status |

---

## Project Structure

```
agent-mini/
├── src/agent_mini/
│   ├── cli.py              # CLI commands
│   ├── config.py           # Config loading
│   ├── bus.py              # Message routing
│   ├── agent/
│   │   ├── loop.py         # Core ReAct agent loop
│   │   ├── context.py      # System prompt builder
│   │   ├── memory.py       # Persistent JSON memory
│   │   └── tools.py        # Built-in tools
│   ├── providers/
│   │   ├── base.py         # Provider interface
│   │   ├── ollama.py       # Ollama
│   │   ├── gemini.py       # Google Gemini
│   │   ├── github_copilot.py  # GitHub Copilot
│   │   └── local.py        # OpenAI-compatible
│   └── channels/
│       ├── base.py         # Channel interface
│       ├── telegram.py     # Telegram bot
├── pyproject.toml
└── config.example.json
```

## Configuration

Full config lives at `~/.agent-mini/config.json`. See [config.example.json](config.example.json) for all options.

Key paths:
- Config: `~/.agent-mini/config.json`
- Workspace: `~/.agent-mini/workspace/`
- Memory: `~/.agent-mini/memory.json`

## License

MIT
