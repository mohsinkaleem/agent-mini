# Agent Mini

**Ultra-lightweight personal AI agent** — inspired by [nanobot](https://github.com/HKUDS/nanobot), built lean.

- ~1,500 lines of Python (core agent)
- 4 LLM providers: **Ollama**, **Gemini**, **GitHub Copilot**, **Local** (any OpenAI-compatible)
- 2 chat channels: **Telegram**, **WhatsApp**
- Built-in tools: shell, files, web search, persistent memory
- Zero-framework: pure `httpx` + `asyncio` — no LangChain, no LiteLLM

## Quick Start

### 1. Install

```bash
# From source (recommended)
git clone https://github.com/your-repo/agent-mini.git
cd agent-mini
pip install -e ".[all]"

# Or with uv
uv pip install -e ".[all]"
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
```

### 4. Gateway (Telegram / WhatsApp)

```bash
agent-mini gateway
```

---

## Providers

Set `"provider"` in config to one of these, then configure its section under `"providers"`:

| Provider | Description | Config key |
|----------|-------------|------------|
| **Ollama** | Local models via Ollama | `ollama` |
| **Gemini** | Google Generative AI | `gemini` |
| **GitHub Copilot** | Copilot chat API (OAuth) | `github_copilot` |
| **Local** | Any OpenAI-compatible endpoint | `local` |

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
      "model": "llama3.1"
    }
  }
}
```

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
      "network": {
        "proxy": "",
        "trustEnv": true,
        "caCertFile": "",
        "compatibilityTls": false,
        "insecureSkipVerify": false,
        "baseUrl": "",
        "baseFileUrl": ""
      }
    }
  }
}
```

4. Run: `agent-mini gateway`

`network` options are optional:
- `proxy`: explicit proxy URL (e.g. `http://127.0.0.1:7890`)
- `trustEnv`: use `HTTP(S)_PROXY` from environment
- `caCertFile`: PEM bundle used for TLS verification (corporate SSL inspection CAs)
- `compatibilityTls`: force conservative TLS 1.2 profile for strict middleboxes
- `insecureSkipVerify`: disable certificate verification (last resort only)
- `baseUrl` / `baseFileUrl`: custom Telegram Bot API endpoint (self-hosted/mirrored)

If you see OpenDNS/Cisco Umbrella errors, your network is intercepting or blocking Telegram. In that case:
1. Prefer an allowed network, VPN, or `network.proxy`
2. Install your corporate root CA if TLS inspection is required
3. Use `insecureSkipVerify: true` only as a temporary workaround

### WhatsApp

Requires **Node.js ≥ 18**.

1. Configure:

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

2. Run: `agent-mini gateway`
3. Scan the QR code that appears in the terminal with WhatsApp → Settings → Linked Devices

---

## Tools

The agent has these built-in tools:

| Tool | Description |
|------|-------------|
| `shell_exec` | Run any shell command |
| `read_file` | Read file contents |
| `write_file` | Create/overwrite files |
| `list_directory` | Browse the filesystem |
| `web_search` | Brave Search (requires API key) |
| `memory_store` | Save info to persistent memory |
| `memory_recall` | Search persistent memory |

### Web Search

Get a free API key at [brave.com/search/api](https://brave.com/search/api/):

```json
{
  "tools": {
    "webSearch": {
      "provider": "brave",
      "apiKey": "BSA..."
    }
  }
}
```

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
| `agent-mini gateway` | Start Telegram/WhatsApp gateway |
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
│       └── whatsapp.py     # WhatsApp bridge client
├── whatsapp-bridge/        # Node.js WhatsApp Web bridge
│   ├── package.json
│   └── index.js
├── pyproject.toml
└── config.example.json
```

## Configuration

Full config lives at `~/.agent-mini/config.json`. See [config.example.json](config.example.json) for all options.

Key paths:
- Config: `~/.agent-mini/config.json`
- Workspace: `~/.agent-mini/workspace/`
- Memory: `~/.agent-mini/memory.json`
- WhatsApp data: `~/.agent-mini/whatsapp-data/`

## License

MIT
