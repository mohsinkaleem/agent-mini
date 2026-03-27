# OpenClaw-Inspired Improvements for agent-mini

> **Date:** 2026-03-09
> **Scope:** Detailed feature & architectural improvements derived from reviewing the [OpenClaw](https://github.com/openclaw/openclaw) codebase and docs, mapped to the agent-mini architecture.
> **Philosophy:** Keep agent-mini lean (~3200 LOC), zero-framework, pure async/httpx. Every improvement is opt-in, not a required dependency.

---

## Table of Contents

1. [Context Window Intelligence](#1-context-window-intelligence)
2. [Provider Failover & Model Management](#2-provider-failover--model-management)
3. [Session Pruning Pipeline](#3-session-pruning-pipeline)
4. [Compaction System](#4-compaction-system)
5. [Memory Architecture Overhaul](#5-memory-architecture-overhaul)
6. [Skills Platform](#6-skills-platform)
7. [System Prompt Engineering](#7-system-prompt-engineering)
8. [Streaming & Chunking Pipeline](#8-streaming--chunking-pipeline)
9. [Plugin Hooks & Lifecycle](#9-plugin-hooks--lifecycle)
10. [Security Hardening](#10-security-hardening)
11. [Session Concurrency & Queueing](#11-session-concurrency--queueing)
12. [Diagnostics & Observability](#12-diagnostics--observability)
13. [Automation (Cron, Webhooks)](#13-automation-cron-webhooks)
14. [Browser Control Tool](#14-browser-control-tool)
15. [Multi-Agent Coordination](#15-multi-agent-coordination)
16. [Implementation Roadmap](#16-implementation-roadmap)

---

## 1. Context Window Intelligence

### What OpenClaw Does

OpenClaw tracks the model's context window per-model from a registry, estimates token usage in real-time, and triggers compaction/pruning automatically when approaching limits. It uses `contextWindow` from the model definition plus a `reserveTokensFloor` to leave headroom.

### Current agent-mini State

- `_summarize_history()` triggers at a fixed **50 messages**, summarizing the oldest 20.
- No token counting — uses message count as a proxy.
- No model-specific context window awareness.
- Summarization is the only context management strategy.

### Proposed Changes

#### 1.1 Token Estimation Engine

Add a lightweight token estimator (no external dependency):

```python
# src/agent_mini/agent/token_estimator.py

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return max(1, len(text) // 4)

def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += 4  # message overhead
        if isinstance(msg.get("content"), str):
            total += estimate_tokens(msg["content"])
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text":
                    total += estimate_tokens(part["text"])
                elif part.get("type") == "image_url":
                    total += 765  # typical image token cost
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += estimate_tokens(json.dumps(tc.get("function", {})))
    return total
```

#### 1.2 Model Context Window Registry

```python
# Add to config.py or a new models_registry.py

MODEL_CONTEXT_WINDOWS = {
    # Ollama models
    "llama3.2:3b": 131072,
    "llama3.1:8b": 131072,
    "qwen2.5:7b": 131072,
    "phi-4-mini": 131072,
    "gemma3:4b": 32768,
    "mistral:7b": 32768,
    # Gemini
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-flash": 1048576,
    # Copilot / OpenAI
    "gpt-4o-mini": 128000,
    "gpt-4.1-mini": 128000,
    # Default fallback
    "_default": 8192,
}

def get_context_window(model_name: str) -> int:
    """Resolve context window for a model, with fallback."""
    for key in [model_name, model_name.split(":")[0], "_default"]:
        if key in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[key]
    return 8192
```

#### 1.3 Token-Aware Summarization Trigger

Replace message-count trigger with token-aware trigger in `loop.py`:

```python
# In AgentLoop.run(), before calling provider:
context_window = get_context_window(self.provider.model_name)
reserve = int(context_window * 0.2)  # 20% headroom for response
current_tokens = estimate_messages_tokens(messages)

if current_tokens > (context_window - reserve):
    await self._compact_context(conversation, context_window)
```

**Files affected:** `agent/loop.py`, new `agent/token_estimator.py`, `config.py`

**Acceptance criteria:**
- Long conversations auto-compact based on token estimate, not message count.
- Models with small context windows (8K) compact more aggressively.
- Models with large windows (128K+) let conversations grow naturally.

---

## 2. Provider Failover & Model Management

### What OpenClaw Does

- **Auth profile rotation**: multiple API keys per provider, round-robin with session stickiness.
- **Exponential backoff cooldowns**: 1m → 5m → 25m → 1h cap after repeated failures.
- **Error taxonomy**: classifies failures as `auth`, `rate_limit`, `billing`, `transient`, `invalid_request`.
- **Model fallbacks**: `agents.defaults.model.fallbacks` list tried in order when all profiles for primary fail.
- **Billing disables**: credit/quota failures get longer cooldowns (5h → 24h).

### Current agent-mini State

- Single active provider.
- Retry with exponential backoff for HTTP 429/500-504.
- No failover, no cooldowns, no error classification.

### Proposed Changes

#### 2.1 Error Classification

```python
# src/agent_mini/providers/base.py

class ProviderErrorKind(Enum):
    AUTH = "auth"            # 401, 403, invalid key
    RATE_LIMIT = "rate_limit"  # 429
    BILLING = "billing"      # insufficient credits
    TRANSIENT = "transient"  # 500-504, network errors
    INVALID = "invalid"      # 400, bad request format
    TIMEOUT = "timeout"      # request timeout

def classify_error(status_code: int, body: str = "") -> ProviderErrorKind:
    if status_code in (401, 403):
        return ProviderErrorKind.AUTH
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status_code == 400:
        if "credit" in body.lower() or "billing" in body.lower() or "quota" in body.lower():
            return ProviderErrorKind.BILLING
        return ProviderErrorKind.INVALID
    if status_code in range(500, 505):
        return ProviderErrorKind.TRANSIENT
    return ProviderErrorKind.TRANSIENT
```

#### 2.2 Provider Cooldown State

```python
@dataclass
class ProviderState:
    name: str
    cooldown_until: float = 0.0
    error_count: int = 0
    last_error_kind: ProviderErrorKind | None = None

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def record_failure(self, kind: ProviderErrorKind):
        self.error_count += 1
        self.last_error_kind = kind
        # Exponential backoff: 60s, 300s, 1500s, 3600s cap
        delay = min(60 * (5 ** (self.error_count - 1)), 3600)
        if kind == ProviderErrorKind.BILLING:
            delay = min(18000 * (2 ** (self.error_count - 1)), 86400)  # 5h → 24h
        self.cooldown_until = time.time() + delay

    def record_success(self):
        self.error_count = 0
        self.cooldown_until = 0.0
```

#### 2.3 Failover Chain Configuration

```json
{
  "provider": "ollama",
  "failover": ["gemini", "local"],
  "providers": {
    "ollama": { "model": "qwen2.5:7b" },
    "gemini": { "model": "gemini-2.0-flash" },
    "local": { "model": "llama-3.2-3b", "baseUrl": "http://localhost:1234/v1" }
  }
}
```

#### 2.4 Failover Logic in Agent Loop

```python
async def _call_with_failover(self, messages, tool_defs, on_stream):
    providers = [self.primary_provider] + self.fallback_providers
    for provider in providers:
        state = self.provider_states[provider.name]
        if not state.is_available:
            continue
        try:
            result = await self._call_provider_with_retry(provider, messages, tool_defs, on_stream)
            state.record_success()
            return result
        except ProviderError as e:
            kind = classify_error(e.status_code, e.body)
            state.record_failure(kind)
            if kind == ProviderErrorKind.INVALID:
                raise  # Don't failover on bad requests
            logger.warning(f"Provider {provider.name} failed ({kind.value}), trying next...")
            continue
    raise AllProvidersExhaustedError("All providers failed or in cooldown")
```

**Files affected:** `providers/base.py`, `agent/loop.py`, `providers/__init__.py`, `config.py`

---

## 3. Session Pruning Pipeline

### What OpenClaw Does

Session pruning trims **old tool results** from the in-memory context before each LLM call. It does NOT rewrite the on-disk session history. Two-stage approach:
- **Soft-trim**: keeps head + tail of large tool outputs, inserts `...` marker.
- **Hard-clear**: replaces entire tool result with a placeholder.

### Current agent-mini State

- Tool outputs are truncated at execution time (50KB shell, 100KB files).
- No post-hoc pruning of accumulated tool results in conversation history.
- Long sessions accumulate tool output that degrades quality.

### Proposed Changes

```python
# src/agent_mini/agent/pruning.py

@dataclass
class PruningConfig:
    enabled: bool = True
    keep_last_assistants: int = 3
    soft_trim_max_chars: int = 4000
    soft_trim_head: int = 1500
    soft_trim_tail: int = 1500
    hard_clear_after_ratio: float = 0.5  # clear if > 50% of context
    min_prunable_chars: int = 5000
    placeholder: str = "[Old tool result cleared to save context space]"

def prune_tool_results(messages: list[dict], config: PruningConfig, context_budget: int) -> list[dict]:
    """Prune old tool results from message list, in-memory only."""
    if not config.enabled:
        return messages

    # Find cutoff: protect messages after the last N assistant messages
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(assistant_indices) < config.keep_last_assistants:
        return messages  # Not enough history to prune

    cutoff = assistant_indices[-config.keep_last_assistants]
    pruned = []

    for i, msg in enumerate(messages):
        if i >= cutoff or msg.get("role") != "tool":
            pruned.append(msg)
            continue

        content = msg.get("content", "")
        if len(content) < config.min_prunable_chars:
            pruned.append(msg)
            continue

        # Soft trim: keep head + tail
        if len(content) < config.soft_trim_max_chars * 2:
            trimmed = (
                content[:config.soft_trim_head]
                + f"\n\n... [{len(content)} chars trimmed] ...\n\n"
                + content[-config.soft_trim_tail:]
            )
            pruned.append({**msg, "content": trimmed})
        else:
            # Hard clear
            pruned.append({**msg, "content": config.placeholder})

    return pruned
```

**Integration point:** Call `prune_tool_results()` in `AgentLoop.run()` just before sending messages to the provider, but never modify the stored conversation.

**Files affected:** New `agent/pruning.py`, `agent/loop.py`, `config.py`

---

## 4. Compaction System

### What OpenClaw Does

- **Auto-compaction**: triggers when approaching context window limit.
- **Manual compaction** via `/compact` with optional instructions.
- **Pre-compaction memory flush**: silent agentic turn to persist important memories before compacting.
- **Compaction model override**: can use a different (stronger) model for summarization.
- **Identifier preservation**: keeps opaque IDs (commit hashes, file paths) in summaries.

### Current agent-mini State

- `_summarize_history()` creates a single flat summary.
- No hierarchical summaries, no compaction model override.
- No pre-compaction memory flush.

### Proposed Changes

#### 4.1 Hierarchical Compaction

Instead of a flat summary, maintain a chain of summaries:

```python
async def _compact_context(self, conversation: list, context_window: int):
    """Hierarchical compaction: summarize → keep recent → chain summaries."""
    # 1. Pre-compaction memory flush (silent)
    if self.memory and self.config.agent.memory_flush_on_compact:
        await self._flush_memories_before_compact(conversation)

    # 2. Find split point: keep last N messages that fit in 40% of context
    target_keep = int(context_window * 0.4)
    keep_from = len(conversation)
    running_tokens = 0
    for i in range(len(conversation) - 1, -1, -1):
        msg_tokens = estimate_tokens(json.dumps(conversation[i]))
        if running_tokens + msg_tokens > target_keep:
            break
        running_tokens += msg_tokens
        keep_from = i

    # 3. Summarize everything before keep_from
    old_messages = conversation[:keep_from]
    if not old_messages:
        return

    # Use compaction model if configured, else primary
    summary = await self._generate_summary(old_messages)

    # 4. Replace: [prior_summary] + [new_summary] + [recent_messages]
    compaction_entry = {
        "role": "system",
        "content": f"<conversation_summary>\n{summary}\n</conversation_summary>"
    }
    conversation[:keep_from] = [compaction_entry]
```

#### 4.2 Compaction Model Override

```json
{
  "agent": {
    "compaction": {
      "model": "ollama/llama3.1:8b",
      "preserveIdentifiers": true,
      "memoryFlushBeforeCompact": true
    }
  }
}
```

#### 4.3 `/compact` Slash Command

Add to CLI:

```python
elif user_input.startswith("/compact"):
    instructions = user_input[8:].strip()
    await agent.compact_now(conversation, instructions)
    console.print("[green]Context compacted.[/green]")
```

**Files affected:** `agent/loop.py`, `cli.py`, `config.py`

---

## 5. Memory Architecture Overhaul

### What OpenClaw Does

- **Plain Markdown memory files**: `MEMORY.md` (curated) + `memory/YYYY-MM-DD.md` (daily logs).
- **Vector memory search**: hybrid BM25 + vector embeddings with MMR re-ranking for diversity.
- **Auto memory flush**: before compaction, prompts model to persist important notes.
- **Memory tools**: `memory_search` (semantic) + `memory_get` (targeted read).
- **Extra indexable paths**: can index external markdown directories.

### Current agent-mini State

- Single `memory.json` file with key-value entries.
- TF-IDF fuzzy search with custom stemmer.
- Max 1000 entries, sliding window.
- No semantic/vector search, no daily logs, no structured memory files.

### Proposed Changes

#### 5.1 Dual-Layer Memory

```
~/.agent-mini/
  memory.json          → existing (keep for backward compat, rename to legacy)
  memory/
    MEMORY.md           → curated long-term knowledge
    2026-03-09.md       → daily notes (auto-created)
    2026-03-08.md
```

#### 5.2 Hybrid Search (TF-IDF + Embedding)

Keep TF-IDF as default (zero-dependency), add optional embedding mode:

```python
class MemorySearchStrategy(Enum):
    TFIDF = "tfidf"          # Default: existing TF-IDF (no dependency)
    EMBEDDING = "embedding"  # Ollama/Gemini embeddings
    HYBRID = "hybrid"        # TF-IDF + embedding combined

class HybridMemorySearch:
    def __init__(self, strategy: MemorySearchStrategy, provider=None):
        self.tfidf = TFIDFSearch()
        self.embedding = EmbeddingSearch(provider) if strategy != MemorySearchStrategy.TFIDF else None

    async def search(self, query: str, top_k: int = 10) -> list[MemoryResult]:
        if self.strategy == MemorySearchStrategy.TFIDF:
            return self.tfidf.search(query, top_k)

        tfidf_results = self.tfidf.search(query, top_k * 2)
        embed_results = await self.embedding.search(query, top_k * 2) if self.embedding else []

        # Merge with weighted scoring
        return self._merge_results(tfidf_results, embed_results, top_k)

    def _merge_results(self, tfidf, embed, top_k, tfidf_weight=0.4, embed_weight=0.6):
        """Weighted merge of TF-IDF and embedding results."""
        scored = {}
        for r in tfidf:
            scored[r.key] = tfidf_weight * r.score
        for r in embed:
            scored.setdefault(r.key, 0.0)
            scored[r.key] += embed_weight * r.score
        # Sort by combined score, return top_k
        ...
```

#### 5.3 Auto-Memory Flush

```python
async def _flush_memories_before_compact(self, conversation):
    """Silent agentic turn: ask model to persist important notes before compaction."""
    flush_prompt = (
        "The conversation is about to be compacted. Review the recent exchanges "
        "and use memory_store to save any important facts, decisions, preferences, "
        "or project context that should be remembered long-term. "
        "Reply with NO_REPLY if nothing needs saving."
    )
    # Run a mini agent turn with just memory tools
    await self._run_silent_turn(conversation, flush_prompt, allowed_tools=["memory_store"])
```

**Files affected:** `agent/memory.py`, `agent/loop.py`, `config.py`

---

## 6. Skills Platform

### What OpenClaw Does

- **SKILL.md format**: markdown files with YAML frontmatter (name, description, metadata).
- **Three locations**: bundled → managed/local → workspace skills (workspace wins).
- **Gating**: `requires.bins`, `requires.env`, `requires.config`, `requires.os` filter at load time.
- **Token-efficient**: only skill names/descriptions injected in system prompt; full instructions loaded on-demand via `read` tool.
- **Session snapshot**: skills cached at session start, hot-reloaded on file change.
- **ClawHub**: public skill registry for discovery and installation.

### Current agent-mini State

- Plugin system: Python `.py` files in `~/.agent-mini/plugins/` with `TOOL_DEF` + `handler`.
- No metadata gating, no on-demand loading, no skill descriptions in prompt.

### Proposed Changes

#### 6.1 Skill Format

```
~/.agent-mini/skills/
  web-researcher/
    SKILL.md
  code-reviewer/
    SKILL.md
  <workspace>/skills/
    project-helper/
      SKILL.md
```

`SKILL.md` format:

```markdown
---
name: web-researcher
description: Deep web research with source verification and citation.
requires:
  env: []
  bins: []
  os: []
enabled: true
---

# Web Researcher Skill

When asked to research a topic deeply, follow this workflow:
1. Use web_search to find 3-5 relevant sources
2. Use web_fetch to read the top results
3. Cross-reference facts across sources
4. Cite sources in your response

## Guidelines
- Always verify claims across multiple sources
- Prefer primary sources over aggregator sites
- Include publication dates for time-sensitive information
```

#### 6.2 Skill Loading & Gating

```python
# src/agent_mini/agent/skills.py

@dataclass
class Skill:
    name: str
    description: str
    path: str
    requires_env: list[str] = field(default_factory=list)
    requires_bins: list[str] = field(default_factory=list)
    requires_os: list[str] = field(default_factory=list)
    enabled: bool = True

def load_skills(workspace: str) -> list[Skill]:
    """Load skills from all locations, workspace wins on conflict."""
    skills = {}
    for location in [BUNDLED_SKILLS, MANAGED_SKILLS, workspace_skills(workspace)]:
        for skill_dir in discover_skill_dirs(location):
            skill = parse_skill_md(skill_dir / "SKILL.md")
            if skill and is_eligible(skill):
                skills[skill.name] = skill  # later location overwrites
    return list(skills.values())

def is_eligible(skill: Skill) -> bool:
    if not skill.enabled:
        return False
    if skill.requires_os and sys.platform not in skill.requires_os:
        return False
    for env_var in skill.requires_env:
        if not os.environ.get(env_var):
            return False
    for binary in skill.requires_bins:
        if not shutil.which(binary):
            return False
    return True
```

#### 6.3 Compact Skill Injection in System Prompt

```python
def format_skills_for_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = ["<available_skills>"]
    for s in skills:
        lines.append(f'  <skill name="{s.name}" location="{s.path}">{s.description}</skill>')
    lines.append("</available_skills>")
    lines.append("\nTo use a skill, read the SKILL.md at its location for full instructions.")
    return "\n".join(lines)
```

This keeps the base prompt small (~100 chars per skill) while enabling targeted skill usage. The full instructions are only loaded when the model reads the SKILL.md file.

**Files affected:** New `agent/skills.py`, `agent/context.py`, `config.py`

---

## 7. System Prompt Engineering

### What OpenClaw Does

- Structured sections: Tooling, Safety, Skills, Workspace, Documentation, Runtime, Time, Heartbeats.
- **Prompt modes**: `full` (main session), `minimal` (sub-agents), `none` (identity only).
- **Bootstrap files**: `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md` injected on turn 1.
- **Truncation with markers**: large files trimmed with `bootstrapMaxChars`.
- **Token budget awareness**: tracks prompt size contributions.

### Current agent-mini State

- Single `build_system_prompt()` in `context.py` with static instructions.
- User system prompt from config appended.
- 5 most recent memories included.
- No sections, no bootstrap files, no prompt modes.

### Proposed Changes

#### 7.1 Sectioned System Prompt

```python
def build_system_prompt(config, memory, skills=None, mode="full") -> str:
    sections = []

    # 1. Identity & Role
    sections.append(build_identity_section(config))

    # 2. Available tools (always)
    sections.append(build_tools_section(config))

    if mode == "full":
        # 3. Safety guardrails
        sections.append(build_safety_section())

        # 4. Skills (if any)
        if skills:
            sections.append(format_skills_for_prompt(skills))

        # 5. Workspace context
        sections.append(f"Working directory: {config.workspace}")

        # 6. Memory context
        recent = memory.get_recent(5) if memory else []
        if recent:
            sections.append(build_memory_section(recent))

        # 7. Runtime info
        sections.append(build_runtime_section(config))

    elif mode == "minimal":
        sections.append(f"Working directory: {config.workspace}")

    return "\n\n---\n\n".join(s for s in sections if s)
```

#### 7.2 Workspace Bootstrap Files

```
~/.agent-mini/workspace/
  AGENTS.md     → operating instructions, coding style
  IDENTITY.md   → agent name, personality, tone
  USER.md       → user profile, preferences
```

These are injected into the first turn of a new session, trimmed to a configurable max:

```python
BOOTSTRAP_FILES = ["AGENTS.md", "IDENTITY.md", "USER.md"]
BOOTSTRAP_MAX_CHARS = 20000  # per file
BOOTSTRAP_TOTAL_MAX = 50000  # total across all files

def inject_bootstrap(workspace: str, conversation: list):
    """Inject workspace bootstrap files into first turn context."""
    if conversation:  # Not first turn
        return
    context_parts = []
    total = 0
    for fname in BOOTSTRAP_FILES:
        path = Path(workspace) / fname
        if path.exists():
            content = path.read_text()[:BOOTSTRAP_MAX_CHARS]
            if total + len(content) > BOOTSTRAP_TOTAL_MAX:
                break
            context_parts.append(f"<{fname}>\n{content}\n</{fname}>")
            total += len(content)
    if context_parts:
        return "\n\n".join(context_parts)
```

**Files affected:** `agent/context.py`, `agent/loop.py`, `config.py`

---

## 8. Streaming & Chunking Pipeline

### What OpenClaw Does

- **Block streaming**: coarse chunks sent as complete channel messages.
- **Preview streaming**: single message updated in-place with latest text.
- **Chunking algorithm**: min/max char bounds, break on paragraph → newline → sentence → whitespace → hard.
- **Code fence handling**: never splits inside fences; closes/reopens if forced.
- **Coalescing**: merges nearby chunks with idle gap detection.
- **Human-like pacing**: random delays between block replies.

### Current agent-mini State

- Telegram: single preview message updated in-place (rate-limited 0.7s, min 80 chars).
- Long responses split at 4000 chars (Telegram limit).
- No block streaming, no coalescing, no code fence awareness.

### Proposed Changes

#### 8.1 Smart Chunking Engine

```python
# src/agent_mini/channels/chunker.py

@dataclass
class ChunkConfig:
    min_chars: int = 200
    max_chars: int = 1200
    break_preference: list[str] = field(default_factory=lambda: ["paragraph", "newline", "sentence", "whitespace"])

class SmartChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self._in_code_fence = False

    def chunk(self, text: str) -> list[str]:
        """Split text into readable chunks, respecting code fences and break preferences."""
        chunks = []
        remaining = text
        while len(remaining) > self.config.min_chars:
            if len(remaining) <= self.config.max_chars:
                chunks.append(remaining)
                remaining = ""
                break
            # Find best break point
            split_at = self._find_break(remaining[:self.config.max_chars])
            chunk = remaining[:split_at]
            # Handle code fence closure
            if self._has_unclosed_fence(chunk):
                chunk += "\n```"
                remaining = "```\n" + remaining[split_at:]
            else:
                remaining = remaining[split_at:]
            chunks.append(chunk.strip())
        if remaining.strip():
            chunks.append(remaining.strip())
        return chunks

    def _find_break(self, text: str) -> int:
        """Find best break point in text using preference order."""
        for break_type in self.config.break_preference:
            pos = self._find_break_of_type(text, break_type)
            if pos and pos >= self.config.min_chars:
                return pos
        return self.config.max_chars  # Hard break

    def _find_break_of_type(self, text: str, break_type: str) -> int | None:
        if break_type == "paragraph":
            pos = text.rfind("\n\n")
        elif break_type == "newline":
            pos = text.rfind("\n")
        elif break_type == "sentence":
            for sep in [". ", "! ", "? "]:
                pos = text.rfind(sep)
                if pos > 0:
                    return pos + len(sep)
            return None
        elif break_type == "whitespace":
            pos = text.rfind(" ")
        else:
            return None
        return pos if pos > 0 else None
```

#### 8.2 Coalescing for Channel Output

```python
class OutputCoalescer:
    """Merge small consecutive chunks before sending to channels."""
    def __init__(self, min_chars: int = 800, idle_ms: int = 1500):
        self.min_chars = min_chars
        self.idle_ms = idle_ms
        self._buffer = ""
        self._last_append = 0

    async def append(self, text: str, send_fn):
        self._buffer += text
        self._last_append = time.time()

        if len(self._buffer) >= self.min_chars:
            await send_fn(self._buffer)
            self._buffer = ""

    async def flush(self, send_fn):
        if self._buffer:
            await send_fn(self._buffer)
            self._buffer = ""
```

**Files affected:** New `channels/chunker.py`, `channels/telegram.py`, `channels/base.py`

---

## 9. Plugin Hooks & Lifecycle

### What OpenClaw Does

- **Hook points**: before_model_resolve, before_prompt_build, before_agent_start, agent_end, before/after_tool_call, tool_result_persist, message_received/sending/sent, session_start/end, before/after_compaction.
- **Plugin registration**: plugins declare hooks they want to intercept.
- **Pipeline**: hooks run in order, can modify params or short-circuit.

### Current agent-mini State

- Plugins export `TOOL_DEF` + `handler`. No lifecycle hooks.
- `on_stream` and `on_tool_event` callbacks exist but are CLI-only.

### Proposed Changes

```python
# src/agent_mini/agent/hooks.py

class HookPoint(Enum):
    BEFORE_PROMPT_BUILD = "before_prompt_build"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_COMPACTION = "before_compaction"
    AGENT_END = "agent_end"
    MESSAGE_RECEIVED = "message_received"

class HookManager:
    def __init__(self):
        self._hooks: dict[HookPoint, list[Callable]] = defaultdict(list)

    def register(self, point: HookPoint, fn: Callable):
        self._hooks[point].append(fn)

    async def run(self, point: HookPoint, context: dict) -> dict:
        for fn in self._hooks[point]:
            result = fn(context) if not asyncio.iscoroutinefunction(fn) else await fn(context)
            if result is not None:
                context = result
        return context
```

Plugin discovery extended:

```python
# In plugin .py files:
TOOL_DEF = { ... }           # Optional: tool definition
HOOKS = {                     # Optional: lifecycle hooks
    "before_tool_call": my_before_tool,
    "after_tool_call": my_after_tool,
}
async def handler(arguments):  # Optional: tool handler
    ...
```

**Files affected:** New `agent/hooks.py`, `agent/tools.py`, `agent/loop.py`

---

## 10. Security Hardening

### What OpenClaw Does

- **DM pairing**: unknown senders get a pairing code; must be approved before agent processes messages.
- **Docker sandboxing**: non-main sessions run in Docker containers.
- **Tool policy**: explicit allow/deny lists per session type.
- **SSRF protection**: browser navigation guarded against private network access.
- **Confirmation prompts**: sensitive macOS operations require TCC-style approval.
- **Threat model atlas**: documented attack vectors and mitigations.

### Current agent-mini State

- Sandbox levels: `workspace`, `readonly` with command blocklist.
- Path restriction to workspace when configured.
- Shell command blocklist (rm -rf, sudo, etc).
- No DM pairing, no Docker sandbox, no SSRF protection, no tool allow/deny.

### Proposed Changes

#### 10.1 Tool Allow/Deny Lists

```json
{
  "tools": {
    "allow": ["read_file", "search_files", "web_search", "memory_recall"],
    "deny": ["shell_exec"],
    "confirmCommands": ["git\\s+push", "pip\\s+install"]
  }
}
```

```python
def get_tool_defs(self) -> list[dict]:
    all_tools = self._all_tool_defs()
    allow = set(self.config.tools.allow) if self.config.tools.allow else None
    deny = set(self.config.tools.deny) if self.config.tools.deny else set()

    filtered = []
    for tool in all_tools:
        name = tool["function"]["name"]
        if allow and name not in allow:
            continue
        if name in deny:
            continue
        filtered.append(tool)
    return filtered
```

#### 10.2 Prompt Injection Detection

```python
# Basic heuristic for detecting prompt injection in tool outputs
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*you\s+must",
    r"<\|system\|>",
    r"\[INST\]",
]

def check_injection(text: str) -> bool:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
```

#### 10.3 DM Pairing for Public Channels

```python
# src/agent_mini/channels/pairing.py

class PairingManager:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._approved: set[str] = set()
        self._pending: dict[str, str] = {}  # user_id -> code
        self._load()

    def is_approved(self, user_id: str) -> bool:
        return user_id in self._approved

    def create_code(self, user_id: str) -> str:
        code = secrets.token_hex(3).upper()  # 6-char hex code
        self._pending[user_id] = code
        self._save()
        return code

    def approve(self, code: str) -> str | None:
        for uid, c in self._pending.items():
            if c == code:
                self._approved.add(uid)
                del self._pending[uid]
                self._save()
                return uid
        return None
```

**Files affected:** `agent/tools.py`, new `channels/pairing.py`, `config.py`

---

## 11. Session Concurrency & Queueing

### What OpenClaw Does

- **Per-session lock**: serializes runs within a session.
- **Global lane**: optional serialization across all sessions.
- **Queue modes**: `collect` (hold messages), `steer` (inject mid-run), `followup` (sequential).
- **Debounce + cap**: incoming messages can be debounced or capped.

### Current agent-mini State

- No session locking. Concurrent requests can corrupt session JSON.
- `MessageBus.handle_message()` can be called concurrently for the same session.

### Proposed Changes

```python
# src/agent_mini/sessions.py

class SessionLock:
    """Per-session asyncio lock to serialize access."""
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def get(cls, session_id: str) -> asyncio.Lock:
        if session_id not in cls._locks:
            cls._locks[session_id] = asyncio.Lock()
        return cls._locks[session_id]

# In bus.py:
async def handle_message(self, channel, user_id, text, stream=False):
    session_key = f"{channel}:{user_id}"
    lock = SessionLock.get(session_key)
    async with lock:
        # Existing logic — no concurrent mutation of same session
        ...
```

#### Queue Mode for Channels

```python
class QueueMode(Enum):
    COLLECT = "collect"    # Hold messages, process after current run
    STEER = "steer"       # Inject into current run after tool calls
    FOLLOWUP = "followup"  # Queue for next run

class SessionQueue:
    def __init__(self, mode: QueueMode = QueueMode.COLLECT):
        self.mode = mode
        self._pending: list[str] = []

    def enqueue(self, message: str):
        self._pending.append(message)

    def drain(self) -> list[str]:
        msgs = self._pending[:]
        self._pending.clear()
        return msgs
```

**Files affected:** `sessions.py`, `bus.py`, `agent/loop.py`

---

## 12. Diagnostics & Observability

### What OpenClaw Does

- **`openclaw doctor`**: validates config, provider health, workspace, channels.
- **`openclaw status`**: shows running state, sessions, token usage.
- **Structured logging**: JSON logs with correlation IDs.
- **Health checks**: periodic provider pings.
- **Turn traces**: per-turn detailed execution records.

### Proposed Changes

#### 12.1 `agent-mini doctor` Command

```python
@cli.command()
def doctor():
    """Diagnose configuration and runtime health."""
    checks = [
        ("Config file", check_config_exists),
        ("Workspace writable", check_workspace_writable),
        ("Provider reachable", check_provider_reachable),
        ("Memory store", check_memory_healthy),
        ("Session store", check_sessions_dir),
        ("Plugin loading", check_plugins_valid),
    ]
    for name, check_fn in checks:
        try:
            ok, detail = check_fn()
            status = "✓" if ok else "✗"
            console.print(f"  {status} {name}: {detail}")
        except Exception as e:
            console.print(f"  ✗ {name}: {e}")
```

#### 12.2 Turn-Level Tracing

```python
@dataclass
class TurnTrace:
    turn_id: str
    timestamp: str
    user_message: str
    iterations: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    latency_ms: float = 0
    provider: str = ""
    model: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
```

Emit traces per turn, persist to `~/.agent-mini/traces/` with configurable retention.

**Files affected:** `cli.py`, new `diagnostics.py`, `agent/loop.py`

---

## 13. Automation (Cron, Webhooks)

### What OpenClaw Does

- **Cron jobs**: recurring scheduled agent tasks.
- **Webhooks**: HTTP endpoints trigger agent actions.
- **Event subscriptions**: Gmail Pub/Sub, file watchers.

### Proposed Changes

#### 13.1 Lightweight Cron Scheduler

```json
{
  "automation": {
    "cron": [
      {
        "schedule": "0 9 * * *",
        "prompt": "Check the weather and summarize my agenda for today.",
        "channel": "telegram",
        "userId": "123456"
      }
    ]
  }
}
```

Use `croniter` (pip dependency) or a simple home-built cron parser for basic patterns.

#### 13.2 Webhook Endpoint

```python
# In gateway mode, add a simple HTTP webhook handler
@app.route("/webhook/{trigger}")
async def handle_webhook(trigger: str, payload: dict):
    config = webhook_configs.get(trigger)
    if not config:
        return {"error": "unknown trigger"}, 404
    prompt = config["prompt"].format(payload=json.dumps(payload))
    await bus.handle_message(config["channel"], config["userId"], prompt)
    return {"status": "triggered"}
```

**Files affected:** New `scheduler.py`, new `webhooks.py`, `cli.py`, `config.py`

---

## 14. Browser Control Tool

### What OpenClaw Does

- Dedicated Chrome/Chromium profile managed by the agent.
- CDP (Chrome DevTools Protocol) for tab control, navigation, screenshots, actions.
- ARIA snapshots for accessibility-based page understanding.
- Multiple profiles (managed, extension relay, remote CDP).
- SSRF protection on navigation.

### Proposed Changes

This is high complexity. Start with a lightweight `browser_open` tool using Playwright (optional dependency):

```python
# Only available when playwright is installed
try:
    from playwright.async_api import async_playwright
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

async def browser_snapshot(url: str) -> str:
    """Open URL in headless browser, return page text content."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=15000)
        text = await page.inner_text("body")
        await browser.close()
        return text[:80000]
```

Add as a gated tool: only available when `playwright` is installed, discovered via the skills/gating system.

**Files affected:** `agent/tools.py`, `config.py`

---

## 15. Multi-Agent Coordination

### What OpenClaw Does

- **`sessions_*` tools**: `sessions_list`, `sessions_history`, `sessions_send` for inter-session communication.
- **Isolated agents**: per-agent workspaces, sessions, and tool configs.
- **Reply-back**: agents can ping-pong messages with acknowledgment.

### Proposed Changes

For agent-mini, start with a simple sub-agent pattern:

```python
async def _spawn_sub_agent(self, task: str, tools: list[str] | None = None) -> str:
    """Run a focused sub-agent with minimal context and specific tools."""
    sub_loop = AgentLoop(
        provider=self.provider,
        config=self.config,
        memory=self.memory,
        prompt_mode="minimal"  # Smaller system prompt
    )
    result = await sub_loop.run(
        user_message=task,
        conversation=[],  # Fresh context
        tool_subset=tools,
    )
    return result.content
```

This allows the main agent to delegate focused tasks (research, summarization, code review) to a sub-agent with clean context.

**Files affected:** `agent/loop.py`, `config.py`

---

## 16. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
| # | Feature | Effort | Impact | Files |
|---|---------|--------|--------|-------|
| 1 | Token estimation engine | Low | High | New `agent/token_estimator.py` |
| 2 | Token-aware compaction trigger | Low | High | `agent/loop.py` |
| 3 | Session pruning (soft/hard) | Medium | High | New `agent/pruning.py`, `agent/loop.py` |
| 4 | Provider error taxonomy | Low | Medium | `providers/base.py` |
| 5 | Session locking | Low | High | `sessions.py`, `bus.py` |

### Phase 2: Reliability (Week 2-3)
| # | Feature | Effort | Impact | Files |
|---|---------|--------|--------|-------|
| 6 | Provider failover chain | Medium | High | `providers/__init__.py`, `agent/loop.py` |
| 7 | Tool allow/deny lists | Low | High | `agent/tools.py`, `config.py` |
| 8 | Hierarchical compaction | Medium | High | `agent/loop.py` |
| 9 | `/compact` command | Low | Medium | `cli.py` |
| 10 | `doctor` command | Low | Medium | `cli.py` |

### Phase 3: Capabilities (Week 3-5)
| # | Feature | Effort | Impact | Files |
|---|---------|--------|--------|-------|
| 11 | Skills platform (SKILL.md) | Medium | High | New `agent/skills.py` |
| 12 | Sectioned system prompt | Medium | High | `agent/context.py` |
| 13 | Workspace bootstrap files | Low | Medium | `agent/context.py`, `agent/loop.py` |
| 14 | Smart chunking engine | Medium | Medium | New `channels/chunker.py` |
| 15 | Plugin hooks & lifecycle | Medium | Medium | New `agent/hooks.py` |

### Phase 4: Advanced (Week 5-8)
| # | Feature | Effort | Impact | Files |
|---|---------|--------|--------|-------|
| 16 | Hybrid memory search | High | Medium | `agent/memory.py` |
| 17 | Auto-memory flush | Medium | Medium | `agent/loop.py`, `agent/memory.py` |
| 18 | Cron scheduler | Medium | High | New `scheduler.py` |
| 19 | Turn-level tracing | Medium | Medium | `agent/loop.py` |
| 20 | Sub-agent spawning | Medium | High | `agent/loop.py` |

### Dependency Graph

```
Token Estimation ──► Token-Aware Compaction ──► Hierarchical Compaction
                                              ├── Session Pruning
                                              └── Auto-Memory Flush

Error Taxonomy ──► Provider Failover ──► Cooldown State

Session Locking ──► Queue Modes

Skills Platform ──► Sectioned System Prompt ──► Workspace Bootstrap

Plugin Hooks ──► Smart Chunking ──► Coalescing
```

---

## Key Design Principles (from OpenClaw)

1. **Prompt is king**: Most improvements work by giving the model better context, not by adding code complexity.
2. **Token budget awareness**: Every feature should know its token cost. Small models have tiny budgets.
3. **Fail gracefully**: Every external dependency (provider, tool, skill) must have a degraded fallback.
4. **In-memory != on-disk**: Pruning and context shaping work in-memory; compaction persists. Keep them separate.
5. **Workspace is the API**: Skills, bootstrap files, memory — all plain files the user can edit directly.
6. **Lazy loading**: Load full skill instructions only when needed, not at prompt assembly time.
7. **Session stickiness**: Pin choices (provider, model, skills) per session to maintain cache coherence.
