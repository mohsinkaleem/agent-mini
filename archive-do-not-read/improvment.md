# Agent Mini — Feature & Reliability Improvements

> Inspired by [OpenClaw](https://github.com/openclaw/openclaw) architecture patterns and cross-referenced against the current agent-mini codebase.

---

## Table of Contents

1. [Architecture & Reliability](#1-architecture--reliability)
2. [Multi-Channel Expansion](#2-multi-channel-expansion)
3. [Security & Sandboxing](#3-security--sandboxing)
4. [Automation & Scheduling](#4-automation--scheduling)
5. [Session & Context Management](#5-session--context-management)
6. [Provider Resilience & Model Management](#6-provider-resilience--model-management)
7. [Tool System Enhancements](#7-tool-system-enhancements)
8. [Memory & Knowledge](#8-memory--knowledge)
9. [Observability & Diagnostics](#9-observability--diagnostics)
10. [Developer Experience](#10-developer-experience)
11. [Voice & Multi-Modal](#11-voice--multi-modal)
12. [Implementation Priority Matrix](#12-implementation-priority-matrix)

---

## 1. Architecture & Reliability

### 1.1 WebSocket Gateway / Control Plane

**What OpenClaw does:** Runs a local WebSocket control plane (`ws://127.0.0.1:18789`) that manages sessions, channels, tools, and events — decoupling the agent runtime from individual channel implementations.

**Current agent-mini state:** The `MessageBus` in `bus.py` is a simple in-memory router. Channels are tightly coupled to the gateway CLI command. There's no persistent control plane.

**Recommendation:**
- Introduce an optional lightweight WebSocket server (using stdlib `asyncio` or `websockets`) that acts as a control plane.
- The control plane should manage: active sessions, tool registry, channel lifecycle, and health status.
- External clients (CLI, web UI, mobile apps) connect to this single endpoint.
- This enables remote access, multiple frontends, and hot-reloading of channels.

**Files affected:** New `src/agent_mini/gateway.py`, modify `bus.py`, `cli.py`

---

### 1.2 Graceful Shutdown & Signal Handling

**What OpenClaw does:** Proper signal handling with graceful shutdown across all channels, pending tool calls, and active sessions.

**Current agent-mini state:** `cli.py` catches `KeyboardInterrupt` in the gateway but there's no coordinated shutdown for in-flight agent runs or tool executions. The `_chat` function has a `finally` block but doesn't save session state on crash.

**Recommendation:**
- Register `SIGTERM`/`SIGINT` handlers that:
  1. Set a cancellation flag on the `AgentLoop`.
  2. Allow current tool executions to finish (with a timeout).
  3. Auto-save the current session state before exit.
  4. Gracefully close all channels and HTTP clients.
- Add a `drain()` method to `AgentLoop` that waits for in-flight tool calls.

**Files affected:** `cli.py`, `agent/loop.py`, `bus.py`

---

### 1.3 Health Checks & Heartbeats

**What OpenClaw does:** Heartbeat mechanisms to detect provider liveness and channel connectivity.

**Current agent-mini state:** No health checking. If Ollama goes down mid-session, the user only discovers it on the next message.

**Recommendation:**
- Add a `health_check()` method to `BaseProvider` that pings the endpoint (e.g., `GET /api/tags` for Ollama, a lightweight Gemini call, etc.).
- Run periodic health checks in gateway mode (every 60s).
- Surface health status in `/status` command and gateway logs.
- Auto-reconnect channels if they disconnect.

**Files affected:** `providers/base.py`, all provider implementations, `cli.py`

---

## 2. Multi-Channel Expansion

### 2.1 Additional Channel Integrations

**What OpenClaw does:** Supports 20+ channels — WhatsApp, Slack, Discord, Signal, IRC, Matrix, Microsoft Teams, Google Chat, LINE, and more.

**Current agent-mini state:** Only Telegram is supported. The `BaseChannel` abstraction is clean but has only one implementation.

**Recommendation (Priority order):**

| Channel | Complexity | Value | Notes |
|---------|-----------|-------|-------|
| **Discord** | Medium | High | Huge dev community, `discord.py` library |
| **Slack** | Medium | High | Enterprise use, Bolt SDK |
| **WhatsApp** | Medium | High | Ubiquitous personal messaging, via Business API |
| **Signal** | Medium | Medium | Privacy-focused, via `signal-cli` |
| **Web Chat** | Low | High | Built-in HTTP/WebSocket chat UI |
| **Matrix** | Medium | Medium | Open protocol, self-hosted option |
| **IRC** | Low | Low | Lightweight, `irc` library |

**Implementation pattern per channel:**
```
src/agent_mini/channels/
    discord.py      # New
    slack.py        # New
    whatsapp.py     # New
    webchat.py      # New (WebSocket-based)
```

**Files affected:** New files under `channels/`, modify `cli.py` gateway, `config.py`

---

### 2.2 Web Chat UI

**What OpenClaw does:** Built-in WebChat channel that provides a browser-based interface.

**Current agent-mini state:** CLI-only or Telegram. No web interface.

**Recommendation:**
- Add a `webchat` channel: a simple async HTTP server (aiohttp or built-in) serving:
  - A static HTML/JS chat page.
  - A WebSocket endpoint for real-time streaming.
- Make it the easiest "first run" experience — `agent-mini serve` opens a browser tab.
- This is the lowest-complexity, highest-value channel addition.

**Files affected:** New `channels/webchat.py`, new `static/` directory, `cli.py`

---

### 2.3 Channel Isolation & Multi-Agent Routing

**What OpenClaw does:** Isolates sessions per channel/account/peer. Group chats get sandboxed Docker environments. DM pairing protocols with allowlisting.

**Current agent-mini state:** `MessageBus` isolates by `channel:user_id` key, but all sessions share the same `AgentLoop` instance (same config, same tools).

**Recommendation:**
- Allow per-channel configuration overrides (e.g., readonly sandbox for group chats, full access for DMs).
- Support per-channel `allowFrom` lists (already exists for Telegram, generalize to all channels).
- Add optional "activation mode" — agent only responds when mentioned (@agent) in group contexts.

**Files affected:** `bus.py`, `config.py`, channel implementations

---

## 3. Security & Sandboxing

### 3.1 Docker Sandbox Mode

**What OpenClaw does:** Runs non-main sessions (group chats, untrusted channels) in Docker containers for isolation. This prevents tool abuse from compromising the host.

**Current agent-mini state:** Sandbox levels (`unrestricted`, `workspace`, `readonly`) and command blocklists exist, but all execution happens on the host.

**Recommendation:**
- Add a `docker` sandbox level that executes `shell_exec` commands inside a Docker container.
- Use a lightweight image (e.g., `python:3.11-slim` or `alpine`) with workspace mounted as a volume.
- Container is created per-session and destroyed on session end.
- Configuration:
  ```json
  {
    "tools": {
      "sandboxLevel": "docker",
      "dockerImage": "agent-mini-sandbox:latest",
      "dockerTimeout": 300
    }
  }
  ```

**Files affected:** `agent/tools.py`, `config.py`, new `sandbox.py`

---

### 3.2 Tool Allowlisting & Denylisting

**What OpenClaw does:** Explicit tool allowlists/denylists per session type. Tools can be toggled on/off granularly.

**Current agent-mini state:** `readonly` mode blocks a fixed set of tools. No per-tool control.

**Recommendation:**
- Add `tools.allow` and `tools.deny` lists in config:
  ```json
  {
    "tools": {
      "allow": ["read_file", "web_search", "memory_recall"],
      "deny": ["shell_exec"]
    }
  }
  ```
- `allow` = whitelist (only these tools available). `deny` = blacklist (everything except these).
- When both are set, `allow` takes precedence.
- Plugin tools should also be subject to these controls.

**Files affected:** `agent/tools.py`, `config.py`

---

### 3.3 Confirmation Prompts for Dangerous Operations

**What OpenClaw does:** Permission prompts before executing sensitive operations (macOS TCC model).

**Current agent-mini state:** Dangerous commands are blocked entirely. No ask-and-confirm flow.

**Recommendation:**
- Add a "confirm" tier between "allow" and "block":
  ```json
  {
    "tools": {
      "confirmCommands": ["\\bgit\\s+push", "\\bpip\\s+install", "\\bnpm\\s+install"]
    }
  }
  ```
- In CLI mode: prompt the user for `[Y/n]` before execution.
- In channel mode: send a confirmation message with inline buttons (Telegram) or reactions (Discord).
- The agent loop pauses and waits for confirmation before proceeding.

**Files affected:** `agent/tools.py`, `agent/loop.py`, `cli.py`, channel implementations

---

## 4. Automation & Scheduling

### 4.1 Cron Jobs & Scheduled Tasks

**What OpenClaw does:** Built-in cron scheduler for recurring agent tasks (e.g., "summarize my emails every morning at 9am").

**Current agent-mini state:** No scheduling capability. The agent only responds to direct messages.

**Recommendation:**
- Add a lightweight scheduler using `asyncio` and cron expressions:
  ```json
  {
    "automation": {
      "cron": [
        {
          "schedule": "0 9 * * *",
          "prompt": "Check the weather and summarize my daily agenda.",
          "channel": "telegram",
          "userId": "123456"
        }
      ]
    }
  }
  ```
- Use `croniter` (lightweight library) or parse simple cron expressions in-house.
- Scheduled prompts run through the same `AgentLoop.run()` path.
- Results are delivered to the specified channel/user.

**Files affected:** New `src/agent_mini/scheduler.py`, `cli.py`, `config.py`

---

### 4.2 Webhook Triggers

**What OpenClaw does:** HTTP webhook endpoints that trigger agent actions when called by external services (GitHub, CI/CD, monitoring, etc.).

**Current agent-mini state:** No webhook support.

**Recommendation:**
- Add a simple HTTP webhook server:
  ```
  POST /webhook/{trigger_name}
  Body: { "payload": "..." }
  ```
- Triggers map to prompts in config:
  ```json
  {
    "automation": {
      "webhooks": {
        "deploy-notify": {
          "prompt": "A deployment just happened: {payload}. Summarize and notify me.",
          "channel": "telegram",
          "userId": "123456"
        }
      }
    }
  }
  ```
- Shared with the gateway HTTP server (or standalone with `--webhooks` flag).

**Files affected:** New `src/agent_mini/webhooks.py`, `cli.py`, `config.py`

---

### 4.3 File Watcher / Event-Driven Actions

**What OpenClaw does:** Node-based device actions and event subscriptions.

**Current agent-mini state:** No event-driven capabilities.

**Recommendation:**
- Add file system watching using `watchdog` or stdlib polling:
  ```json
  {
    "automation": {
      "watch": [
        {
          "path": "~/Downloads",
          "pattern": "*.pdf",
          "prompt": "A new PDF was downloaded: {file}. Summarize its contents."
        }
      ]
    }
  }
  ```
- Lower priority than cron/webhooks, but high value for personal assistant use cases.

**Files affected:** New `src/agent_mini/watcher.py`, `config.py`

---

## 5. Session & Context Management

### 5.1 Smarter Context Window Management

**What OpenClaw does:** Sophisticated session management with multiple summarization strategies and context-aware trimming.

**Current agent-mini state:** `_summarize_history()` triggers at 50 messages, always summarizing the oldest 20. This is simplistic — it can lose critical context and the threshold is arbitrary.

**Recommendation:**
- **Token-aware trimming:** Count tokens (estimated or actual) instead of message count. Trigger summarization when approaching the model's context window (e.g., 80% of limit).
- **Sliding window with pinned messages:** Allow certain messages to be "pinned" (never summarized away).
- **Hierarchical summarization:** Instead of one flat summary, maintain a chain:
  ```
  [Oldest summary] → [Recent summary] → [Last N messages]
  ```
- **Configurable strategy:**
  ```json
  {
    "agent": {
      "contextStrategy": "sliding_window",
      "maxContextTokens": 8000,
      "summaryRetention": 3
    }
  }
  ```

**Files affected:** `agent/loop.py`, `config.py`, new `agent/context_manager.py`

---

### 5.2 Session Metadata & Tagging

**What OpenClaw does:** Rich session metadata including titles, tags, and searchable history.

**Current agent-mini state:** Sessions have basic metadata (id, updated, message count). No titles, tags, or search.

**Recommendation:**
- Auto-generate session titles from the first user message or via LLM.
- Add `/tag` command to label sessions.
- Add `/search` command to search across all session content.
- Session metadata:
  ```json
  {
    "id": "20260307_143022",
    "title": "Fix React auth bug",
    "tags": ["coding", "react"],
    "updated": "2026-03-07T14:30:22",
    "tokenCount": 12500
  }
  ```

**Files affected:** `sessions.py`, `cli.py`

---

### 5.3 Session Branching & Forking

**What OpenClaw does:** Session isolation per channel/peer with independent conversation threads.

**Current agent-mini state:** Sessions are linear. No way to branch or fork a conversation.

**Recommendation:**
- Add `/fork` command that creates a new session from the current one's history.
- Useful for "what if" explorations without losing the original conversation.
- Session storage already supports this — just needs a `parent_id` field.

**Files affected:** `sessions.py`, `cli.py`

---

## 6. Provider Resilience & Model Management

### 6.1 Provider Failover Chain

**What OpenClaw does:** Configurable model selection with automatic failover — if the primary model/provider is unavailable, transparently falls back to alternatives.

**Current agent-mini state:** Single active provider with retry logic for transient HTTP errors, but no failover to a different provider.

**Recommendation:**
- Add a `failover` configuration:
  ```json
  {
    "provider": "gemini",
    "failover": ["local", "ollama"],
    "failoverTimeout": 10
  }
  ```
- When the primary provider fails after retries, automatically try the next in the failover chain.
- Log the failover event and notify the user once.
- Maintain separate health status per provider.

**Files affected:** `providers/__init__.py`, `agent/loop.py`, `config.py`

---

### 6.2 Model Rotation / Load Balancing

**What OpenClaw does:** Model rotation for distributing load and managing rate limits.

**Current agent-mini state:** One model at a time. No rotation.

**Recommendation:**
- For rate-limited APIs (Gemini free tier, Copilot), support round-robin or weighted rotation:
  ```json
  {
    "provider": "gemini",
    "providers": {
      "gemini": {
        "models": [
          {"model": "gemini-2.0-flash", "weight": 3},
          {"model": "gemini-1.5-flash", "weight": 1}
        ]
      }
    }
  }
  ```
- Simple weighted random selection per request.

**Files affected:** `providers/__init__.py`, provider implementations, `config.py`

---

### 6.3 Token Budget & Cost Tracking

**What OpenClaw does:** Detailed usage tracking with cost awareness.

**Current agent-mini state:** Token tracking exists per-turn and per-session, but no cost estimation, no budgets, and no persistent tracking across sessions.

**Recommendation:**
- Add a cost estimation table per provider/model.
- Persist token usage history to `~/.agent-mini/usage.json`.
- Add configurable budgets:
  ```json
  {
    "agent": {
      "maxTokensPerTurn": 10000,
      "maxTokensPerSession": 100000,
      "dailyTokenBudget": 500000
    }
  }
  ```
- Warn when approaching limits. Hard-stop when exceeded.
- `/usage` command to show daily/weekly/monthly stats.

**Files affected:** New `src/agent_mini/usage.py`, `agent/loop.py`, `cli.py`, `config.py`

---

## 7. Tool System Enhancements

### 7.1 Tool Execution Timeout Configuration

**What OpenClaw does:** Configurable timeouts per tool type.

**Current agent-mini state:** `shell_exec` has a hardcoded 120s timeout. `web_fetch` uses 20s. No per-tool configuration.

**Recommendation:**
- Add configurable timeouts:
  ```json
  {
    "tools": {
      "timeouts": {
        "shell_exec": 300,
        "web_fetch": 30,
        "default": 60
      }
    }
  }
  ```

**Files affected:** `agent/tools.py`, `config.py`

---

### 7.2 Parallel Tool Execution Improvements

**What OpenClaw does:** Sophisticated parallel execution with dependency awareness.

**Current agent-mini state:** All tool calls in a single LLM response are executed concurrently via `asyncio.gather()`. No dependency handling, no rate limiting.

**Recommendation:**
- Add concurrency limits (semaphore) to prevent overwhelming the system:
  ```json
  {
    "tools": {
      "maxParallelTools": 5
    }
  }
  ```
- Add a tool execution queue with priority (e.g., `shell_exec` lower priority than `read_file`).
- Consider tool dependencies — if one tool produces output another needs, detect this and serialize.

**Files affected:** `agent/loop.py`, `agent/tools.py`, `config.py`

---

### 7.3 MCP (Model Context Protocol) Integration

**What OpenClaw does:** Likely evolving toward MCP-compatible tool serving.

**Current agent-mini state:** Custom plugin system loading Python files from `~/.agent-mini/plugins/`.

**Recommendation:**
- Add MCP client support so agent-mini can connect to MCP tool servers:
  ```json
  {
    "mcp": {
      "servers": [
        {"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /home/user"},
        {"name": "github", "url": "http://localhost:3001"}
      ]
    }
  }
  ```
- Keep the existing plugin system as-is (it's simple and works).
- MCP tools appear alongside built-in tools in the tool registry.
- This opens the door to the entire MCP ecosystem.

**Files affected:** New `src/agent_mini/mcp_client.py`, `agent/tools.py`, `config.py`

---

### 7.4 Tool Result Caching

**What OpenClaw does:** Intelligent caching to avoid redundant tool calls.

**Current agent-mini state:** No caching. Every `web_search` or `read_file` call executes fresh.

**Recommendation:**
- Add a simple TTL-based in-memory cache for idempotent tools:
  - `web_search(query)` → cache for 15 minutes.
  - `read_file(path)` → cache until file mtime changes.
  - `web_fetch(url)` → cache for 10 minutes.
- Never cache: `shell_exec`, `write_file`, `code_edit`, `memory_store`.
- Configuration:
  ```json
  {
    "tools": {
      "cacheEnabled": true,
      "cacheTTL": 900
    }
  }
  ```

**Files affected:** `agent/tools.py`, `config.py`

---

## 8. Memory & Knowledge

### 8.1 Embedding-Based Memory Search

**What OpenClaw does:** Advanced knowledge retrieval capabilities.

**Current agent-mini state:** TF-IDF with a custom Porter stemmer in `memory.py`. Works well for exact/keyword matches but misses semantic similarity.

**Recommendation:**
- Add optional embedding-based recall using the active LLM provider:
  - Ollama: `POST /api/embeddings` (free, local).
  - Gemini: embedding API.
  - Fallback: keep TF-IDF (zero dependency).
- Store embeddings alongside memory entries.
- Use cosine similarity for recall.
- Configuration:
  ```json
  {
    "memory": {
      "searchStrategy": "embedding",
      "embeddingModel": "nomic-embed-text"
    }
  }
  ```

**Files affected:** `agent/memory.py`, `config.py`

---

### 8.2 Workspace-Aware Memory

**What OpenClaw does:** Workspace-scoped context and configuration.

**Current agent-mini state:** Global memory (`~/.agent-mini/memory.json`). No workspace-scoped memory.

**Recommendation:**
- Add per-workspace memory stored alongside the workspace:
  ```
  ~/.agent-mini/workspace/.memory.json    # Default workspace
  /projects/my-app/.agent-mini/memory.json  # Project-specific
  ```
- Auto-detect workspace from `cwd` or config.
- Global memory for user preferences, workspace memory for project context.

**Files affected:** `agent/memory.py`, `config.py`, `agent/context.py`

---

### 8.3 Auto-Memory (Proactive Learning)

**What OpenClaw does:** Context retention across conversations.

**Current agent-mini state:** Memory is only stored when the agent explicitly calls `memory_store`. No automatic learning.

**Recommendation:**
- At the end of each session (or every N turns), prompt the LLM to extract key facts worth remembering:
  ```
  "Review this conversation and list any facts about the user,
   their preferences, or project context worth remembering."
  ```
- Store extracted facts automatically.
- Configurable: `"memory": { "autoLearn": true, "autoLearnInterval": 10 }`

**Files affected:** `agent/loop.py`, `agent/memory.py`, `config.py`

---

## 9. Observability & Diagnostics

### 9.1 Structured Logging

**What OpenClaw does:** Comprehensive logging and diagnostic tools.

**Current agent-mini state:** Python `logging` module with basic format. No structured output.

**Recommendation:**
- Add JSON structured logging option for production/gateway mode:
  ```json
  {
    "logging": {
      "format": "json",
      "level": "info",
      "file": "~/.agent-mini/logs/agent.log"
    }
  }
  ```
- Include context in every log entry: session_id, channel, user_id, tool_name, latency.
- Add log rotation (daily or size-based).

**Files affected:** `cli.py`, new `src/agent_mini/logging_config.py`

---

### 9.2 Turn-Level Diagnostics

**What OpenClaw does:** Detailed diagnostics per interaction.

**Current agent-mini state:** Tool events are emitted via callbacks but not persisted. Token usage is tracked in memory only.

**Recommendation:**
- Record full turn traces:
  ```json
  {
    "turn_id": "abc123",
    "timestamp": "2026-03-08T10:30:00",
    "user_message": "Fix the bug in auth.py",
    "iterations": 3,
    "tool_calls": [
      {"name": "read_file", "args": {"path": "auth.py"}, "latency_ms": 12, "result_bytes": 3400},
      {"name": "code_edit", "args": {"path": "auth.py", "..."}, "latency_ms": 8}
    ],
    "tokens": {"prompt": 2100, "completion": 450},
    "total_latency_ms": 3200
  }
  ```
- Persist to `~/.agent-mini/traces/` for debugging.
- Add `/trace` command to show last turn's detailed execution path.

**Files affected:** `agent/loop.py`, new `src/agent_mini/tracing.py`, `cli.py`

---

### 9.3 Metrics & Dashboard

**What OpenClaw does:** Monitoring capabilities for long-running gateway deployments.

**Current agent-mini state:** No metrics collection.

**Recommendation:**
- Expose optional Prometheus-compatible metrics endpoint in gateway mode:
  - `agent_requests_total`
  - `agent_tool_calls_total{tool="shell_exec"}`
  - `agent_tokens_used_total{type="prompt"}`
  - `agent_latency_seconds{operation="llm_call"}`
- Lightweight: just a `/metrics` HTTP endpoint alongside the gateway.

**Files affected:** New `src/agent_mini/metrics.py`, `cli.py`

---

## 10. Developer Experience

### 10.1 Hot-Reload Configuration

**What OpenClaw does:** Dynamic configuration without restart.

**Current agent-mini state:** Config is loaded once at startup. The `/model` command can switch providers, but other config changes require restart.

**Recommendation:**
- Watch `config.json` for changes (using file mtime polling, zero dependencies).
- Hot-reload: tools config, sandbox level, blocked commands, system prompt.
- Don't hot-reload: provider (requires re-creation), channel tokens.
- Emit a notification when config is reloaded.

**Files affected:** `config.py`, `agent/tools.py`, `cli.py`

---

### 10.2 Plugin Improvements

**What OpenClaw does:** Skills platform with bundled, managed, and custom workspace skills.

**Current agent-mini state:** Simple plugin system loading from `~/.agent-mini/plugins/`. Plugins are discovered at startup only.

**Recommendation:**
- **Hot-reload plugins** — detect new/changed plugin files without restart.
- **Plugin marketplace/registry** — `agent-mini install <plugin>` to fetch community plugins from a central registry (GitHub-based).
- **Plugin dependencies** — allow plugins to declare pip dependencies in metadata.
- **Plugin lifecycle hooks** — `on_load()`, `on_unload()`, `on_session_start()`.
- **Plugin configuration** — plugins can declare config options accessible via `config.json`.

**Files affected:** `agent/tools.py`, `cli.py`, new `src/agent_mini/plugins.py`

---

### 10.3 REPL Improvements

**What OpenClaw does:** Rich interactive experience with canvas, overlays, etc.

**Current agent-mini state:** Basic Rich-powered REPL with markdown rendering and multi-line input.

**Recommendation:**
- **Tab completion** for slash commands and file paths.
- **History navigation** with up/down arrows (using `prompt_toolkit` or similar).
- **Inline tool progress** — show a progress bar for long tool executions.
- **Conversation streaming with thinking display** — show the model's thinking process (already supported by providers, not surfaced in CLI).
- **Image rendering** — display images inline in supported terminals (iTerm2, Kitty).

**Files affected:** `cli.py`

---

## 11. Voice & Multi-Modal

### 11.1 Voice Input/Output

**What OpenClaw does:** Voice Wake on macOS/iOS, Talk Mode on Android, TTS via ElevenLabs with system fallbacks.

**Current agent-mini state:** Vision support (images → models) exists. No voice capabilities.

**Recommendation:**
- **TTS output:** Add optional text-to-speech using system commands (`say` on macOS, `espeak` on Linux) or ElevenLabs API.
- **STT input:** Add optional speech-to-text using Whisper (via Ollama or OpenAI API).
- Configuration:
  ```json
  {
    "voice": {
      "ttsEnabled": false,
      "ttsProvider": "system",
      "sttEnabled": false,
      "sttModel": "whisper"
    }
  }
  ```
- Start with system-native TTS (zero dependency) and expand later.

**Files affected:** New `src/agent_mini/voice.py`, `cli.py`, `config.py`

---

### 11.2 Enhanced Vision Pipeline

**What OpenClaw does:** Live Canvas with A2UI (agent-driven visual workspace).

**Current agent-mini state:** Image detection from message text, base64 encoding, multi-part content building. Works but basic.

**Recommendation:**
- **Screenshot tool** — add a `screenshot` tool that captures screen or window (using `screencapture` on macOS, `scrot` on Linux).
- **Clipboard image support** — detect and read images from clipboard.
- **PDF/document vision** — extract pages from PDFs as images for analysis.
- **Image generation** — integrate with DALL-E, Stable Diffusion, or Flux for image creation.

**Files affected:** `agent/vision.py`, `agent/tools.py`

---

## 12. Implementation Priority Matrix

| Priority | Feature | Effort | Impact | Category |
|----------|---------|--------|--------|----------|
| **P0** | Provider Failover Chain | Low | High | Reliability |
| **P0** | Graceful Shutdown & Session Save | Low | High | Reliability |
| **P0** | Tool Allowlist/Denylist | Low | High | Security |
| **P0** | Token-Aware Context Management | Medium | High | Reliability |
| **P1** | Web Chat UI Channel | Medium | High | Channels |
| **P1** | Health Checks | Low | Medium | Reliability |
| **P1** | Cron Scheduler | Medium | High | Automation |
| **P1** | Structured Logging | Low | Medium | Observability |
| **P1** | Tool Timeout Configuration | Low | Medium | Tools |
| **P1** | Tool Result Caching | Low | Medium | Tools |
| **P1** | Auto-Memory | Medium | High | Memory |
| **P2** | Discord Channel | Medium | High | Channels |
| **P2** | Slack Channel | Medium | High | Channels |
| **P2** | Webhook Triggers | Medium | Medium | Automation |
| **P2** | Docker Sandbox | Medium | High | Security |
| **P2** | MCP Client Integration | High | High | Tools |
| **P2** | Embedding Memory Search | Medium | Medium | Memory |
| **P2** | Turn-Level Diagnostics | Medium | Medium | Observability |
| **P2** | Confirmation Prompts | Medium | Medium | Security |
| **P2** | Token Budget & Cost Tracking | Medium | Medium | Provider |
| **P3** | Plugin Hot-Reload | Low | Medium | DX |
| **P3** | REPL Improvements | Medium | Medium | DX |
| **P3** | Model Rotation | Low | Low | Provider |
| **P3** | Session Tagging & Search | Low | Low | Sessions |
| **P3** | Workspace-Aware Memory | Low | Medium | Memory |
| **P3** | Hot-Reload Config | Low | Medium | DX |
| **P3** | File Watcher Triggers | Medium | Low | Automation |
| **P3** | Voice Input/Output | High | Medium | Multi-Modal |
| **P3** | WebSocket Control Plane | High | Medium | Architecture |
| **P3** | Metrics & Dashboard | Medium | Low | Observability |
| **P3** | Screenshot Tool | Low | Low | Multi-Modal |

---

## Quick Wins (Can Be Done in a Day Each)

1. **Provider Failover** — Wrap `_call_provider_with_retry` to try failover providers.
2. **Graceful Shutdown** — Add signal handlers and auto-save in `cli.py`.
3. **Tool Allow/Deny Lists** — Filter in `get_tool_defs()` and `execute()`.
4. **Health Checks** — Add `health_check()` to providers with simple ping.
5. **Configurable Timeouts** — Read timeouts from config in `tools.py`.
6. **Tool Result Caching** — Simple dict cache with TTL in `ToolExecutor`.
7. **Structured Logging** — JSON formatter option in `_setup_logging()`.

---

## Summary

Agent-mini has a solid, clean architecture with well-separated concerns. The biggest opportunities inspired by OpenClaw are:

1. **Reliability** — Failover chains, health checks, and smarter context management will make the agent production-ready.
2. **Multi-channel** — A web chat UI and Discord/Slack channels would dramatically expand the user base.
3. **Automation** — Cron and webhooks turn agent-mini from a reactive chatbot into a proactive personal assistant.
4. **Security** — Docker sandboxing and granular tool control are essential for any shared/remote deployment.
5. **Observability** — Structured logging and tracing are table stakes for debugging agent behavior.

The lean, zero-framework philosophy should be preserved — each improvement should be an opt-in enhancement, not a required dependency.
