# Agent Mini

**Ultra-lightweight personal AI agent** — inspired by [nanobot](https://github.com/HKUDS/nanobot), built lean.

- ~1,500 lines of Python (core agent)
- 4 LLM providers: **Ollama**, **Gemini**, **GitHub Copilot**, **Local** (any OpenAI-compatible)
- 2 chat channels: **Telegram**, **WhatsApp**
- Built-in tools: shell, files, web search & browse, persistent memory
- Zero-framework: pure `httpx` + `asyncio` — no LangChain, no LiteLLM
- **Free web search** — DuckDuckGo scraping, no API key required

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

`streamResponses: true` enables real-time streamed Telegram replies (native with Ollama; other providers fall back to full-response mode).

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

### WhatsApp Bridge: Build + Packaging

What it does:
- `whatsapp-bridge/index.js` is a Node process that connects to WhatsApp Web, receives incoming messages, and forwards them to the Python agent webhook.
- It also exposes `POST /send` so Python can send replies back to WhatsApp chats.

How it runs by default:
- No manual JS build is required for normal usage.
- On first WhatsApp gateway start, Agent Mini runs `npm install` in the bridge directory if `node_modules` is missing.
- Then it launches the bridge with `node index.js`.

Packaging behavior:
- Python wheel includes bridge source files (`index.js`, `package.json`, `package-lock.json`).
- `node_modules` are **not** bundled, so package size stays small.

Optional build step (for maintainers):
```bash
cd whatsapp-bridge
npm install
npm run build
```
- This creates a bundled bridge file at `whatsapp-bridge/dist/index.cjs`.
- Runtime prefers `dist/index.cjs` when present, otherwise uses `index.js`.

---

## Tools

The agent has these built-in tools (all available out of the box — no API keys needed):

| Tool | Description |
|------|-------------|
| `shell_exec` | Run any shell command |
| `read_file` | Read file contents |
| `append_file` | Append content to a file |
| `write_file` | Create/overwrite files |
| `list_directory` | Browse the filesystem |
| `search_files` | Search text/regex across files (uses `rg` or `grep`) |
| `web_search` | Search the web via DuckDuckGo (free, no API key) |
| `web_fetch` | Fetch & read any web page as plain text |
| `memory_store` | Save info to persistent memory |
| `memory_recall` | Search persistent memory |

### Web Search & Browsing

Web search uses **DuckDuckGo HTML scraping** — free with no API key needed. It works out of the box.

The agent can:
1. **Search** — `web_search("Python asyncio tutorial")` returns titles, URLs, and snippets
2. **Read** — `web_fetch("https://example.com/article")` fetches a page and extracts readable text

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
