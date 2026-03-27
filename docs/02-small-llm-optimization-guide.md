# Harnessing Small LLMs: Maximizing Agent Performance

> **Date:** 2026-03-09
> **Context:** agent-mini is designed to work with small/local LLMs (1B–14B parameters) via Ollama, LM Studio, llama.cpp, etc. This document covers battle-tested techniques for squeezing maximum capability, accuracy, and reliability from these models in an agentic context.
> **Audience:** Developers building agent-mini features and choosing prompt/architecture strategies for small models.

---

## Table of Contents

1. [The Small Model Landscape](#1-the-small-model-landscape)
2. [System Prompt Engineering](#2-system-prompt-engineering)
3. [Tool Call Optimization](#3-tool-call-optimization)
4. [Context Window Management](#4-context-window-management)
5. [Structured Output Strategies](#5-structured-output-strategies)
6. [Multi-Turn Conversation Techniques](#6-multi-turn-conversation-techniques)
7. [Error Recovery & Self-Correction](#7-error-recovery--self-correction)
8. [Memory & Retrieval for Small Models](#8-memory--retrieval-for-small-models)
9. [Task Decomposition & Chaining](#9-task-decomposition--chaining)
10. [Model Selection Strategy](#10-model-selection-strategy)
11. [Quantization & Inference Optimization](#11-quantization--inference-optimization)
12. [Prompt Templates & Few-Shot Patterns](#12-prompt-templates--few-shot-patterns)
13. [Evaluation & Benchmarking](#13-evaluation--benchmarking)
14. [Architecture Patterns for Small Models](#14-architecture-patterns-for-small-models)
15. [Concrete Implementation Plan](#15-concrete-implementation-plan)

---

## 1. The Small Model Landscape

### Model Tiers (as of March 2026)

| Tier | Params | Examples | Context | Tool Calling | Best For |
|------|--------|---------|---------|--------------|----------|
| **Tiny** | 1-3B | Llama 3.2 1B/3B, Phi-4 Mini, Qwen2.5 1.5B/3B, Gemma 3 1B | 32K-128K | Basic | Simple Q&A, summarization |
| **Small** | 4-8B | Llama 3.1/3.3 8B, Qwen2.5 7B, Gemma 3 4B, Mistral 7B v0.3, DeepSeek-R1-Distill-Qwen-7B | 32K-128K | Good | General assistant, coding, analysis |
| **Medium** | 9-14B | Qwen2.5 14B, Gemma 3 12B, Mistral Nemo 12B, Phi-4 14B | 32K-128K | Strong | Complex reasoning, multi-step tasks |
| **Edge Cloud** | Free API | Gemini 2.0 Flash, GPT-4.1 Mini (Copilot), DeepSeek V3 (free tier) | 128K-1M | Excellent | Best quality when internet available |

### Key Constraints of Small Models

1. **Limited reasoning depth**: Can follow 3-5 step chains reliably; struggles beyond 7+ steps.
2. **Instruction following degrades with prompt length**: Performance drops sharply when system prompt exceeds ~2000 tokens.
3. **Tool call format fragility**: Smaller models are more likely to malform JSON tool calls.
4. **Context window ≠ effective context**: A 128K context model may lose accuracy after ~8K tokens of actual use.
5. **Hallucination rate scales inversely with parameter count**: More aggressive grounding needed.
6. **Single-task focus**: Performs best when given one clear objective per turn, not multi-objective instructions.

---

## 2. System Prompt Engineering

The system prompt is the single highest-leverage optimization for small models. Every token matters.

### 2.1 Principles

| Principle | Why It Matters for Small Models |
|-----------|-------------------------------|
| **Brevity** | Small models lose instruction-following ability as prompt grows. Keep under 800 tokens. |
| **Structure** | XML tags and clear section headers improve parsing accuracy by 15-30%. |
| **Specificity** | "Read the file first" > "Consider reading the file if needed". Direct commands, not suggestions. |
| **Constraint-first** | State what NOT to do before what to do. Small models anchor on early instructions. |
| **Examples > rules** | One concrete example teaches more than three abstract rules. |

### 2.2 Optimized System Prompt Template

```
You are a coding assistant with access to tools. You operate in a workspace directory.

<rules>
- ALWAYS read a file before editing it
- NEVER guess file contents — use read_file or search_files first
- Run ONE tool at a time unless explicitly asked for parallel execution
- If a tool fails, try a different approach — do not retry the same call
- Keep responses concise: answer the question, nothing more
</rules>

<workspace>{workspace_path}</workspace>

<available_tools>
{tool_list_compact}
</available_tools>
```

**Size budget**: ~400 tokens for system prompt, leaving maximum room for conversation.

### 2.3 Dynamic Prompt Scaling

Adjust system prompt complexity based on model capability:

```python
def build_system_prompt_for_model(config, memory, model_name: str) -> str:
    model_tier = classify_model_tier(model_name)

    if model_tier == "tiny":  # 1-3B
        # Minimal prompt: just rules and tools
        return MINIMAL_SYSTEM_PROMPT.format(
            workspace=config.workspace,
            tool_list=format_tool_list_compact(config),
        )
    elif model_tier == "small":  # 4-8B
        # Standard prompt: rules, tools, brief memory
        return STANDARD_SYSTEM_PROMPT.format(
            workspace=config.workspace,
            tool_list=format_tool_list_compact(config),
            memory=format_recent_memory(memory, max_entries=3),
        )
    else:  # medium or cloud
        # Full prompt: rules, tools, memory, skills, bootstrap
        return FULL_SYSTEM_PROMPT.format(
            workspace=config.workspace,
            tool_list=format_tool_list_full(config),
            memory=format_recent_memory(memory, max_entries=5),
            skills=format_skills_compact(config),
        )
```

### 2.4 Tool Descriptions: Short > Long

Small models parse shorter tool descriptions more reliably.

**Bad (too verbose):**
```json
{
  "name": "read_file",
  "description": "Read the contents of a file at the specified path. The file must exist in the workspace directory. Returns the text content of the file. You should use this tool whenever you need to examine a file's contents before making changes to it."
}
```

**Good (concise):**
```json
{
  "name": "read_file",
  "description": "Read a file. Args: path (relative to workspace). Returns file text content."
}
```

**Benchmark impact**: Concise descriptions reduce tool selection errors by ~20% on 7B models.

### 2.5 Instruction Anchoring

Place the most important instructions at the **beginning** and **end** of the system prompt (primacy/recency effect is stronger in small models):

```
<critical>
ALWAYS read before editing. NEVER guess file contents.
</critical>

... (rest of prompt) ...

<reminder>
Read files before editing. Use tools to verify — don't assume.
</reminder>
```

---

## 3. Tool Call Optimization

### 3.1 Reduce Tool Count

Small models degrade significantly with more than 8-10 tool definitions. Strategy: **tiered tool exposure**.

```python
# Tier 1 (always available, max 8): core tools only
TIER1_TOOLS = [
    "read_file", "write_file", "code_edit",
    "shell_exec", "search_files", "list_directory",
    "web_search", "memory_recall"
]

# Tier 2 (available on demand): extended tools
TIER2_TOOLS = [
    "web_fetch", "append_file", "memory_store",
    "screenshot", "browser_open"
]

def get_tool_defs_for_model(model_tier: str) -> list[dict]:
    if model_tier in ("tiny", "small"):
        return [t for t in ALL_TOOLS if t["function"]["name"] in TIER1_TOOLS]
    return ALL_TOOLS
```

### 3.2 Simplify Tool Schemas

Small models handle flat, simple schemas better than nested ones.

**Bad (nested, complex):**
```json
{
  "name": "code_edit",
  "parameters": {
    "type": "object",
    "properties": {
      "file": {"type": "object", "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}}},
      "edit": {"type": "object", "properties": {"old_text": {"type": "string"}, "new_text": {"type": "string"}}}
    }
  }
}
```

**Good (flat):**
```json
{
  "name": "code_edit",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "File path"},
      "old_text": {"type": "string", "description": "Exact text to replace"},
      "new_text": {"type": "string", "description": "Replacement text"}
    },
    "required": ["path", "old_text", "new_text"]
  }
}
```

### 3.3 Tool Call Validation & Repair

Small models frequently produce malformed tool calls. Add a repair layer:

```python
async def _execute_tool_calls_with_repair(self, tool_calls: list[ToolCall]) -> list[dict]:
    results = []
    for tc in tool_calls:
        try:
            # 1. Validate JSON arguments
            args = self._parse_and_repair_args(tc.function.arguments)

            # 2. Validate required parameters
            missing = self._check_required_params(tc.function.name, args)
            if missing:
                results.append({
                    "role": "tool",
                    "content": f"Error: missing required parameters: {missing}. "
                               f"Required: {self._get_required_params(tc.function.name)}",
                    "tool_call_id": tc.id,
                })
                continue

            # 3. Execute
            result = await self.executor.execute(tc.function.name, args)
            results.append(result)

        except json.JSONDecodeError:
            # 4. Attempt JSON repair for common small-model errors
            repaired = self._repair_json(tc.function.arguments)
            if repaired:
                result = await self.executor.execute(tc.function.name, repaired)
                results.append(result)
            else:
                results.append({
                    "role": "tool",
                    "content": f"Error: malformed JSON arguments. Expected valid JSON.",
                    "tool_call_id": tc.id,
                })
    return results

def _repair_json(self, raw: str) -> dict | None:
    """Attempt common JSON repairs for small model outputs."""
    s = raw.strip()
    # Fix trailing commas
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    # Fix unquoted keys
    s = re.sub(r'(\w+)\s*:', r'"\1":', s)
    # Fix single quotes
    s = s.replace("'", '"')
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None
```

### 3.4 Sequential vs Parallel Execution

Small models make more errors when asked to produce multiple tool calls simultaneously. Default to **sequential** for small models:

```python
# In loop.py
if model_tier in ("tiny", "small") and len(tool_calls) > 1:
    # Execute sequentially — small models may have dependency errors in parallel calls
    for tc in tool_calls:
        result = await self._execute_single_tool(tc)
        # Feed result back before next tool (gives model a chance to correct course)
        ...
else:
    # Parallel execution for capable models
    results = await asyncio.gather(*[self._execute_single_tool(tc) for tc in tool_calls])
```

### 3.5 Tool Output Compression

Small models have limited context. Compress tool outputs aggressively:

```python
def compress_tool_output(output: str, model_tier: str) -> str:
    """Compress tool output based on model capacity."""
    limits = {
        "tiny": 2000,    # 1-3B: very aggressive
        "small": 5000,   # 4-8B: moderate
        "medium": 15000, # 9-14B: generous
        "cloud": 50000,  # API models: full output
    }
    max_chars = limits.get(model_tier, 10000)

    if len(output) <= max_chars:
        return output

    # Smart truncation: keep head + tail
    head = max_chars * 2 // 3
    tail = max_chars // 3
    return (
        output[:head]
        + f"\n\n... [{len(output) - max_chars} chars truncated] ...\n\n"
        + output[-tail:]
    )
```

---

## 4. Context Window Management

### 4.1 Effective Context vs Advertised Context

Research consistently shows that small models' performance degrades well before their advertised context limit:

| Model Size | Advertised Context | Effective Context (90% accuracy) |
|------------|-------------------|----------------------------------|
| 1-3B | 32K-128K | ~2K-4K tokens |
| 4-8B | 32K-128K | ~4K-8K tokens |
| 9-14B | 32K-128K | ~8K-16K tokens |
| Cloud (flash) | 128K-1M | ~32K-64K tokens |

**Implication**: Target **effective context**, not advertised. This changes compaction/pruning thresholds dramatically.

### 4.2 Aggressive Context Budget

```python
# Effective context budgets by model tier
EFFECTIVE_CONTEXT = {
    "tiny": 3000,    # ~12KB of text
    "small": 6000,   # ~24KB of text
    "medium": 12000, # ~48KB of text
    "cloud": 32000,  # ~128KB of text
}

def get_effective_context(model_name: str) -> int:
    tier = classify_model_tier(model_name)
    return EFFECTIVE_CONTEXT.get(tier, 8000)
```

### 4.3 Context Allocation Strategy

Divide the context budget into zones:

```
┌─────────────────────────────────────────────┐
│ System Prompt          │ ~400-800 tokens     │  (10-15%)
├────────────────────────┤                     │
│ Compaction Summary     │ ~500-1000 tokens    │  (10-15%)
├────────────────────────┤                     │
│ Recent Conversation    │ ~2000-4000 tokens   │  (50-60%)
│ (user + assistant +    │                     │
│  tool results)         │                     │
├────────────────────────┤                     │
│ Response Headroom      │ ~1000-2000 tokens   │  (20-25%)
└─────────────────────────────────────────────┘
```

```python
@dataclass
class ContextBudget:
    system_prompt: int
    compaction_summary: int
    conversation: int
    response_headroom: int

    @classmethod
    def for_model(cls, model_name: str) -> "ContextBudget":
        total = get_effective_context(model_name)
        return cls(
            system_prompt=int(total * 0.12),
            compaction_summary=int(total * 0.13),
            conversation=int(total * 0.55),
            response_headroom=int(total * 0.20),
        )
```

### 4.4 Sliding Window with Priority

Not all messages are equal. Assign priorities:

```python
class MessagePriority(Enum):
    CRITICAL = 3   # System prompt, compaction summary
    HIGH = 2       # Last user message, last assistant response
    MEDIUM = 1     # Recent tool results, conversation
    LOW = 0        # Old tool results, old exchanges

def prioritize_messages(messages: list[dict]) -> list[tuple[dict, int]]:
    """Assign priority scores for context window fitting."""
    result = []
    n = len(messages)
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            priority = MessagePriority.CRITICAL
        elif i >= n - 2:  # Last exchange
            priority = MessagePriority.HIGH
        elif msg["role"] == "tool" and i < n - 6:
            priority = MessagePriority.LOW
        else:
            priority = MessagePriority.MEDIUM
        result.append((msg, priority.value))
    return result

def fit_to_budget(messages: list[dict], budget: int) -> list[dict]:
    """Keep highest-priority messages within budget."""
    prioritized = prioritize_messages(messages)
    # Always keep system + last exchange
    critical = [(m, p) for m, p in prioritized if p >= 2]
    optional = [(m, p) for m, p in prioritized if p < 2]

    # Sort optional by priority (high first), then recency
    optional.sort(key=lambda x: (-x[1], -messages.index(x[0])))

    result = [m for m, _ in critical]
    remaining = budget - sum(estimate_tokens(json.dumps(m)) for m in result)

    for msg, _ in optional:
        cost = estimate_tokens(json.dumps(msg))
        if remaining - cost < 0:
            continue
        result.append(msg)
        remaining -= cost

    # Restore original order
    return sorted(result, key=lambda m: messages.index(m))
```

### 4.5 Compaction Thresholds by Model Tier

```python
def should_compact(messages: list[dict], model_name: str) -> bool:
    tier = classify_model_tier(model_name)
    budget = ContextBudget.for_model(model_name)
    current = estimate_messages_tokens(messages)
    return current > (budget.system_prompt + budget.compaction_summary + budget.conversation)
```

---

## 5. Structured Output Strategies

### 5.1 XML Tags for Structured Thinking

Small models benefit from explicit structure markers:

```
When solving a problem, use this format:

<think>
Brief analysis of what needs to be done
</think>

<plan>
1. First step
2. Second step
</plan>

<action>
[tool call or response]
</action>
```

This improves step-following accuracy by ~25% on 7B models compared to free-form output.

### 5.2 Constrained Generation with Grammars

When using Ollama or llama.cpp, leverage grammar-based constrained generation:

```python
# For Ollama provider, add grammar support
async def chat(self, messages, tools, temperature):
    payload = {
        "model": self.model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        # Use Ollama's native tool calling mode
        payload["tools"] = tools
    elif self._require_json:
        # Constrain to JSON output
        payload["format"] = "json"
    ...
```

### 5.3 Response Format Hints

For small models, add explicit format hints in the user message:

```python
def augment_user_message(message: str, model_tier: str) -> str:
    """Add format hints for small models to improve output quality."""
    if model_tier in ("tiny", "small"):
        # Add grounding reminder
        return (
            f"{message}\n\n"
            "(Remember: read files before editing. Use tools to verify facts. "
            "Be concise in your response.)"
        )
    return message
```

---

## 6. Multi-Turn Conversation Techniques

### 6.1 Conversation Shaping

Small models easily lose track of the conversation goal. Re-inject the user's original request:

```python
def inject_goal_reminder(conversation: list, original_goal: str) -> list:
    """For small models: periodically re-inject the user's goal."""
    if len(conversation) > 10 and estimate_messages_tokens(conversation) > 3000:
        reminder = {
            "role": "system",
            "content": f"Reminder: the user's original request was: {original_goal}"
        }
        # Insert before the last assistant message
        conversation.insert(-1, reminder)
    return conversation
```

### 6.2 Conversation Summarization Prompt

Tailor the summarization prompt for small models:

```python
COMPACT_SUMMARY_PROMPT = """Summarize this conversation in under 200 words.
Focus on: (1) what the user asked, (2) what was done, (3) what's left to do.
Keep file paths, error messages, and code snippets verbatim.
Output only the summary, nothing else."""
```

Large models can handle open-ended summarization. Small models need tight constraints.

### 6.3 Reduce Iteration Ceiling

Small models are more likely to enter loops or degrade after many iterations:

```python
MAX_ITERATIONS = {
    "tiny": 5,     # Very limited
    "small": 10,   # Moderate
    "medium": 15,  # Standard
    "cloud": 25,   # Extended
}
```

---

## 7. Error Recovery & Self-Correction

### 7.1 Structured Error Feedback

OpenClaw-style: wrap errors with nudge phrases. For small models, be more explicit:

```python
def format_tool_error(tool_name: str, error: str, model_tier: str) -> str:
    """Format error message to guide small model recovery."""
    if model_tier in ("tiny", "small"):
        return (
            f"Tool '{tool_name}' failed: {error}\n\n"
            f"What to do next:\n"
            f"- If file not found: use search_files or list_directory to find the correct path\n"
            f"- If permission denied: check if the path is in the workspace\n"
            f"- If command failed: try a simpler command or different approach\n"
            f"DO NOT retry the exact same call."
        )
    else:
        return f"Tool '{tool_name}' failed: {error}. Try a different approach."
```

### 7.2 Loop Detection

Small models frequently repeat failed tool calls. Detect and break loops:

```python
class LoopDetector:
    def __init__(self, max_repeats: int = 2):
        self.max_repeats = max_repeats
        self._recent_calls: list[str] = []

    def record(self, tool_name: str, args_hash: str) -> bool:
        """Record a tool call. Returns True if this is a repeated loop."""
        key = f"{tool_name}:{args_hash}"
        self._recent_calls.append(key)

        # Check for repeats in the last N calls
        recent = self._recent_calls[-6:]
        count = recent.count(key)
        return count > self.max_repeats

    def get_loop_breaker_message(self) -> str:
        return (
            "You are repeating the same tool call that already failed. "
            "STOP and try a completely different approach. "
            "If you cannot proceed, explain what you're stuck on."
        )
```

### 7.3 Graceful Degradation

When a small model produces unusable output (no tool calls AND no meaningful text), fall back:

```python
async def _handle_empty_response(self, conversation, iteration):
    """Handle cases where small model produces empty/useless output."""
    if iteration > 2:
        # Model is stuck — inject a rescue prompt
        conversation.append({
            "role": "user",
            "content": (
                "You seem to be stuck. Please either:\n"
                "1. Use a tool to take the next step, OR\n"
                "2. Explain what you need to proceed, OR\n"
                "3. Provide your best answer with what you know."
            )
        })
        return True  # Continue loop
    return False
```

---

## 8. Memory & Retrieval for Small Models

### 8.1 Retrieval-Augmented Context

For small models, memory recall is critical — they can't maintain long-term context internally. Automatically inject relevant memories:

```python
async def _build_messages_with_rag(self, user_message: str, conversation: list) -> list:
    """Augment context with relevant memories for small models."""
    messages = list(conversation)

    # 1. Search memory for relevant context
    if self.memory:
        relevant = self.memory.recall(user_message)
        if relevant and relevant != "No matching memories":
            rag_context = {
                "role": "system",
                "content": f"<relevant_context>\n{relevant}\n</relevant_context>"
            }
            # Insert after system prompt, before conversation
            insert_pos = 1 if messages and messages[0]["role"] == "system" else 0
            messages.insert(insert_pos, rag_context)

    return messages
```

### 8.2 Memory Importance Scoring

Not all memories are equal. Score by relevance + recency + importance:

```python
def score_memory(entry: dict, query: str, now: float) -> float:
    """Combined relevance score for memory entries."""
    # TF-IDF relevance
    relevance = tfidf_score(query, entry["value"])

    # Recency decay: half-life of 7 days
    age_days = (now - entry["timestamp"]) / 86400
    recency = 0.5 ** (age_days / 7)

    # Importance: user-pinned memories get a boost
    importance = 1.5 if entry.get("pinned") else 1.0

    return relevance * 0.6 + recency * 0.3 + importance * 0.1
```

### 8.3 Workspace Grounding

For coding tasks, automatically provide workspace context:

```python
async def _ground_with_workspace(self, user_message: str) -> str:
    """Add workspace file listing for grounding small models."""
    if any(kw in user_message.lower() for kw in ["file", "code", "edit", "fix", "bug", "implement"]):
        listing = await self.executor.execute("list_directory", {"path": "."})
        return f"Current workspace files:\n{listing}\n\nUser request: {user_message}"
    return user_message
```

---

## 9. Task Decomposition & Chaining

### 9.1 Automatic Task Decomposition

Complex tasks overwhelm small models. Break them down automatically:

```python
DECOMPOSITION_PROMPT = """Break this task into 2-4 simple steps.
Each step should be one clear action.
Format:
STEP 1: [action]
STEP 2: [action]
...

Task: {task}"""

async def decompose_task(self, task: str, model_tier: str) -> list[str]:
    """For small models: break complex tasks into simple steps."""
    if model_tier not in ("tiny", "small"):
        return [task]  # Large models handle complex tasks directly

    # Check if task seems complex (heuristic)
    complexity_signals = ["and then", "also", "additionally", "after that", "finally",
                          "first", "second", "multiple", "several", "all the"]
    is_complex = any(s in task.lower() for s in complexity_signals) or len(task) > 200

    if not is_complex:
        return [task]

    # Ask the model to decompose
    response = await self._call_provider_with_retry(
        [{"role": "user", "content": DECOMPOSITION_PROMPT.format(task=task)}],
        tool_defs=[],
        on_stream=None,
    )

    steps = [line.split(":", 1)[1].strip()
             for line in response.content.split("\n")
             if line.strip().startswith("STEP")]
    return steps if steps else [task]
```

### 9.2 Step-by-Step Execution

Execute decomposed steps sequentially, feeding each result into the next:

```python
async def run_decomposed(self, steps: list[str], conversation: list, **kwargs):
    """Execute decomposed steps sequentially."""
    for i, step in enumerate(steps):
        step_message = f"Step {i+1}/{len(steps)}: {step}"
        result = await self.run(step_message, conversation, **kwargs)
        # Each step's result is in the conversation for the next step
    return result
```

### 9.3 Verification Steps

After task completion, add a verification step for small models:

```python
VERIFY_PROMPT = (
    "Review what you just did. Did it fully address the original request? "
    "If something is missing or wrong, fix it now. "
    "If everything looks correct, provide a brief summary."
)

async def run_with_verification(self, message, conversation, **kwargs):
    result = await self.run(message, conversation, **kwargs)
    if classify_model_tier(self.provider.model_name) in ("tiny", "small"):
        # Add verification step
        verify_result = await self.run(VERIFY_PROMPT, conversation, **kwargs)
        return verify_result
    return result
```

---

## 10. Model Selection Strategy

### 10.1 Task-Based Model Routing

Use different models for different task types:

```python
TASK_PATTERNS = {
    "summarize": {"tier": "tiny", "suggested": "llama3.2:3b"},
    "translate": {"tier": "tiny", "suggested": "qwen2.5:3b"},
    "code_edit": {"tier": "small", "suggested": "qwen2.5-coder:7b"},
    "research": {"tier": "medium", "suggested": "qwen2.5:14b"},
    "complex_reasoning": {"tier": "cloud", "suggested": "gemini-2.0-flash"},
    "general": {"tier": "small", "suggested": "llama3.1:8b"},
}

def select_model_for_task(task: str, available_models: list[str]) -> str:
    """Route tasks to the most appropriate model."""
    task_lower = task.lower()
    for pattern, config in TASK_PATTERNS.items():
        if pattern in task_lower:
            if config["suggested"] in available_models:
                return config["suggested"]
    return available_models[0]  # Default to primary
```

### 10.2 Quality-Aware Escalation

Start with a small model, escalate to a larger one if quality is poor:

```python
class QualityEscalator:
    """Detect low-quality responses and escalate to a better model."""

    def should_escalate(self, response: str, tool_calls: list, iteration: int) -> bool:
        signals = 0
        # Empty or very short response
        if not response or len(response) < 20:
            signals += 1
        # Repeated failed tool calls
        if iteration > 3 and not tool_calls:
            signals += 1
        # Incoherent response (heuristic)
        if response and self._incoherence_score(response) > 0.5:
            signals += 1
        return signals >= 2

    def _incoherence_score(self, text: str) -> float:
        """Simple heuristic: repeated sentence fragments, garbled output."""
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.0
        # Check for repeated chunks
        seen = set()
        repeats = 0
        for s in sentences:
            s = s.strip().lower()
            if s in seen and len(s) > 20:
                repeats += 1
            seen.add(s)
        return repeats / max(len(sentences), 1)
```

### 10.3 Recommended Model Configurations

```json
{
  "provider": "ollama",
  "failover": ["gemini", "local"],
  "providers": {
    "ollama": {
      "model": "qwen2.5:7b",
      "baseUrl": "http://localhost:11434"
    },
    "gemini": {
      "model": "gemini-2.0-flash",
      "apiKey": "..."
    },
    "local": {
      "model": "llama-3.1-8b",
      "baseUrl": "http://localhost:1234/v1"
    }
  },
  "agent": {
    "compaction": {
      "model": "ollama/qwen2.5:7b"
    }
  }
}
```

**Best small models for agentic use (March 2026):**

| Use Case | Recommended | Why |
|----------|------------|-----|
| **General coding agent** | Qwen2.5-Coder 7B, Qwen2.5 7B | Best tool-calling accuracy at 7B |
| **Fastest responses** | Llama 3.2 3B, Gemma 3 4B | Sub-second inference on CPU |
| **Complex reasoning** | DeepSeek-R1-Distill-Qwen-7B, Phi-4 14B | Best reasoning chains |
| **Free cloud fallback** | Gemini 2.0 Flash | 1M context, excellent tool use, generous free tier |
| **Local + private** | Mistral 7B v0.3, Llama 3.1 8B | Well-tested, no telemetry |
| **Summarization/memory** | Llama 3.2 3B, Qwen2.5 3B | Fast + cheap for background tasks |

---

## 11. Quantization & Inference Optimization

### 11.1 Quantization Sweet Spots

| Quantization | Size Reduction | Quality Loss | Best For |
|-------------|---------------|-------------|----------|
| **Q8_0** | 50% | ~0% | When RAM allows; best quality |
| **Q6_K** | 55% | ~1% | Sweet spot for 7B on 16GB RAM |
| **Q5_K_M** | 60% | ~2% | Good balance for 14B models |
| **Q4_K_M** | 65% | ~3-5% | Run 14B on 8GB RAM; noticeable for tool calls |
| **Q3_K_M** | 75% | ~8-15% | Emergency tier; tool calling degrades |
| **IQ2_XXS** | 85% | ~25%+ | Not recommended for agentic use |

**Recommendation for agent-mini:**
- 7B models: Q6_K or Q5_K_M
- 14B models: Q5_K_M or Q4_K_M
- Use Q8_0 if GPU VRAM allows

### 11.2 Ollama Configuration for Agent Tasks

```
# Optimal Ollama settings for agentic use

# For 7B models (16GB RAM):
ollama run qwen2.5:7b-instruct-q6_K --num-ctx 8192 --num-gpu 99

# For 14B models (16GB RAM):
ollama run qwen2.5:14b-instruct-q4_K_M --num-ctx 4096 --num-gpu 99

# For 3B models (8GB RAM):
ollama run llama3.2:3b-instruct-q8_0 --num-ctx 16384 --num-gpu 99
```

Key parameters:
- `--num-ctx`: Set to your effective context budget, not max. Lower = faster.
- `--num-gpu 99`: Offload all layers to GPU.
- `--num-thread`: Set to physical core count (not hyperthreads).

### 11.3 llama.cpp / LM Studio Tuning

For llama.cpp server or LM Studio, these settings optimize agent performance:

```bash
# Start llama.cpp server optimized for tool calling
./llama-server \
  -m qwen2.5-7b-instruct-q6_K.gguf \
  --ctx-size 8192 \
  --n-predict 2048 \
  --threads 8 \
  --n-gpu-layers 99 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn \
  --mlock \
  --port 1234
```

### 11.4 Speculative Decoding

For setups with both a small and larger model, speculative decoding can speed inference:

```python
# When using Ollama with speculative decoding enabled
# This is transparent to agent-mini but worth configuring in Ollama

# In Ollama Modelfile:
# FROM qwen2.5:14b
# PARAMETER num_predict 2048
# PARAMETER speculative.model qwen2.5:3b
# PARAMETER speculative.num_draft 5
```

---

## 12. Prompt Templates & Few-Shot Patterns

### 12.1 Few-Shot Examples for Tool Calling

Include one example in the system prompt to dramatically improve tool call accuracy:

```python
TOOL_EXAMPLE = """
Example interaction:
User: "What's in the README?"
Assistant: I'll read the README file.
[calls read_file with path="README.md"]
Tool result: "# My Project\nA sample project..."
Assistant: The README contains a project description for "My Project"...
"""
```

**Impact**: Few-shot examples reduce tool call format errors by ~40% on 3-7B models.

### 12.2 Task-Specific Prompt Templates

Pre-built templates for common agent tasks:

```python
TEMPLATES = {
    "code_review": (
        "Review the code changes. Focus on bugs, security issues, and maintainability. "
        "Read each changed file, then provide feedback in a structured format."
    ),
    "bug_fix": (
        "Debug and fix the reported issue. "
        "Steps: 1) Read the relevant file, 2) Identify the bug, "
        "3) Apply the fix using code_edit, 4) Verify the fix is correct."
    ),
    "research": (
        "Research the topic thoroughly. "
        "Use web_search for 2-3 queries, web_fetch for the best results, "
        "then synthesize findings with citations."
    ),
}
```

### 12.3 Chain-of-Thought Prompting

For reasoning tasks, explicitly request step-by-step thinking:

```python
def add_cot_wrapper(message: str, model_tier: str) -> str:
    """Add chain-of-thought wrapper for small models."""
    if model_tier in ("tiny", "small"):
        return (
            f"{message}\n\n"
            "Think through this step by step:\n"
            "1. What do I need to know?\n"
            "2. What tools should I use?\n"
            "3. What's the answer?"
        )
    return message
```

---

## 13. Evaluation & Benchmarking

### 13.1 Agent Accuracy Metrics

Track these metrics to compare model performance:

```python
@dataclass
class AgentMetrics:
    # Tool calling
    tool_call_success_rate: float = 0.0       # Valid tool calls / total tool calls
    tool_call_format_errors: int = 0          # Malformed JSON
    tool_call_wrong_tool: int = 0             # Called wrong tool for the task

    # Task completion
    task_completion_rate: float = 0.0         # Fully completed tasks / total tasks
    average_iterations: float = 0.0           # Turns to complete a task
    loop_detection_count: int = 0             # Times loop detector fired

    # Context efficiency
    tokens_per_task: int = 0                  # Average tokens consumed per task
    compaction_triggers: int = 0              # Times compaction was needed

    # Quality
    self_correction_count: int = 0            # Times model fixed its own error
    escalation_count: int = 0                 # Times escalated to better model
```

### 13.2 Benchmark Suite

Create a simple benchmark for comparing models:

```python
BENCHMARK_TASKS = [
    {
        "name": "simple_read",
        "prompt": "What's in the README.md file?",
        "expected_tools": ["read_file"],
        "difficulty": "easy",
    },
    {
        "name": "code_edit",
        "prompt": "Add a docstring to the main function in app.py",
        "expected_tools": ["read_file", "code_edit"],
        "difficulty": "medium",
    },
    {
        "name": "multi_step",
        "prompt": "Find all TODO comments in the project and create a summary file",
        "expected_tools": ["search_files", "write_file"],
        "difficulty": "hard",
    },
    {
        "name": "research",
        "prompt": "Search the web for the latest Python 3.13 features and summarize them",
        "expected_tools": ["web_search", "web_fetch"],
        "difficulty": "medium",
    },
]
```

### 13.3 A/B Testing Framework

```python
async def benchmark_model(model_name: str, tasks: list[dict]) -> dict:
    """Run benchmark tasks against a model and collect metrics."""
    results = {}
    for task in tasks:
        start = time.time()
        try:
            response = await run_task(model_name, task["prompt"])
            elapsed = time.time() - start
            results[task["name"]] = {
                "success": validate_response(response, task),
                "latency_ms": elapsed * 1000,
                "iterations": response.iterations,
                "tokens": response.total_tokens,
                "tool_calls": [tc.name for tc in response.tool_calls],
            }
        except Exception as e:
            results[task["name"]] = {"success": False, "error": str(e)}
    return results
```

---

## 14. Architecture Patterns for Small Models

### 14.1 The "Narrow Expert" Pattern

Instead of one general agent, create focused sub-agents:

```
User Message
     │
     ▼
┌──────────────┐
│   Router     │  (tiny model: classify intent)
│  (1-3B)      │
└──────┬───────┘
       │
       ├──► Code Agent (7B coder model, code tools only)
       ├──► Research Agent (7B general, web tools only)
       ├──► Memory Agent (3B, memory tools only)
       └──► Chat Agent (7B general, no tools)
```

Each sub-agent gets a minimal, focused prompt and only the tools it needs.

```python
class AgentRouter:
    INTENTS = {
        "code": {"model": "qwen2.5-coder:7b", "tools": ["read_file", "write_file", "code_edit", "shell_exec", "search_files"]},
        "research": {"model": "qwen2.5:7b", "tools": ["web_search", "web_fetch", "memory_store"]},
        "memory": {"model": "llama3.2:3b", "tools": ["memory_store", "memory_recall"]},
        "chat": {"model": "llama3.1:8b", "tools": []},
    }

    async def route(self, message: str) -> str:
        """Classify intent with tiny model, then dispatch to expert."""
        intent = await self._classify(message)
        config = self.INTENTS.get(intent, self.INTENTS["chat"])
        return await self._run_expert(message, config)
```

### 14.2 The "Verify-then-Execute" Pattern

For critical operations, use a two-model approach:

```
User: "Delete all test files"
     │
     ▼
┌──────────────┐     ┌──────────────┐
│ Planning      │────►│ Verification │
│ Model (7B)    │     │ Model (3B)   │
│               │     │              │
│ "I'll delete  │     │ "This will   │
│  tests/*.py"  │     │  delete 12   │
│               │     │  files. OK?" │
└──────────────┘     └──────┬───────┘
                            │
                     User confirms
                            │
                     ┌──────▼───────┐
                     │  Execution   │
                     │  Model (7B)  │
                     └──────────────┘
```

### 14.3 The "Background Preprocessor" Pattern

Use a tiny model for preprocessing that improves the main model's performance:

```python
class BackgroundPreprocessor:
    """Use a fast tiny model for preprocessing tasks."""

    async def preprocess_workspace(self, workspace: str) -> str:
        """Generate workspace summary for context."""
        # Use 3B model to summarize workspace structure
        listing = await list_directory_recursive(workspace, max_depth=2)
        summary = await tiny_model.generate(
            f"Summarize this project structure in 3 sentences:\n{listing}"
        )
        return summary

    async def preprocess_file(self, content: str) -> str:
        """Summarize large files for context injection."""
        if len(content) > 5000:
            summary = await tiny_model.generate(
                f"Summarize the key functions and classes in this code:\n{content[:10000]}"
            )
            return summary
        return content
```

### 14.4 The "Checkpoint & Resume" Pattern

For long tasks, save intermediate state so the model can resume without replaying everything:

```python
@dataclass
class TaskCheckpoint:
    task_id: str
    original_goal: str
    completed_steps: list[str]
    current_step: str
    workspace_state: dict  # key file hashes
    created: float

    def to_context(self) -> str:
        return (
            f"Task: {self.original_goal}\n"
            f"Completed: {', '.join(self.completed_steps)}\n"
            f"Current step: {self.current_step}\n"
            f"Resume from here."
        )
```

---

## 15. Concrete Implementation Plan

### Immediate Wins (1-2 days each)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 1 | **Reduce system prompt to <800 tokens** | +15-20% accuracy for small models | Low |
| 2 | **Tool output compression by model tier** | More context for conversation | Low |
| 3 | **Tool call JSON repair** | -30% tool call failures | Low |
| 4 | **Loop detection** | Prevents infinite loops with small models | Low |
| 5 | **Max iterations by model tier** | Prevents budget waste | Low |
| 6 | **Concise tool descriptions** | -20% tool selection errors | Low |

### Short-Term (1-2 weeks)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 7 | **Token-aware context management** | Proper context utilization | Medium |
| 8 | **Model tier classification** | Enables all per-tier optimizations | Low |
| 9 | **Structured error feedback** | Better self-correction | Low |
| 10 | **Tiered tool exposure** | Better tool selection for small models | Medium |
| 11 | **Task decomposition** | Handle complex tasks with small models | Medium |
| 12 | **Goal reminder injection** | Prevents goal drift in long conversations | Low |

### Medium-Term (2-4 weeks)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 13 | **Quality escalation** | Automatic fallback to better model | Medium |
| 14 | **RAG-enhanced context** | Better grounding for small models | Medium |
| 15 | **Few-shot examples in prompt** | Dramatic tool call improvement | Low |
| 16 | **Agent router (narrow expert)** | Best-in-class per-task performance | High |
| 17 | **Benchmark suite** | Objective model comparison | Medium |
| 18 | **Background preprocessor** | Better workspace awareness | Medium |

### Configuration Recommendation

Add these to `config.py`:

```python
@dataclass
class SmallModelConfig:
    """Configuration for small model optimizations."""
    auto_detect_tier: bool = True          # Classify model tier automatically
    max_tools_per_tier: dict = field(default_factory=lambda: {
        "tiny": 6, "small": 8, "medium": 12, "cloud": 20
    })
    compress_tool_output: bool = True       # Tier-aware output compression
    repair_tool_calls: bool = True          # JSON repair layer
    loop_detection: bool = True             # Break repeated tool call loops
    task_decomposition: bool = True         # Auto-decompose complex tasks
    goal_reminders: bool = True             # Re-inject user goals
    quality_escalation: bool = False        # Auto-escalate to better model
    few_shot_examples: bool = True          # Include tool call examples
    effective_context_override: int = 0     # Override effective context (0 = auto)
```

---

## Key Takeaways

1. **System prompt is the #1 lever**: Under 800 tokens, structured with XML, instruction-anchored. More impactful than any code change.

2. **Context window math is different for small models**: Use 10-20% of the advertised context window as your effective budget. Compact early and often.

3. **Tool count matters more than tool quality**: 6-8 well-described tools beats 15 mediocre ones. Tiered exposure is essential.

4. **Repair > Reject**: Small models produce ~20-30% malformed tool calls. Repairing JSON is cheaper than retrying the whole turn.

5. **Sequential beats parallel for small models**: Feed tool results one at a time. Let the model course-correct between steps.

6. **Decompose everything**: Complex tasks should be broken into 2-4 simple steps. Each step gets clean context.

7. **Error feedback must be prescriptive**: Don't just say "error". Say what went wrong AND what to do next.

8. **Model routing pays off**: Use a tiny model (1-3B) for routing, a medium model (7B) for execution, and escalate to cloud for failures. This is cheaper and more reliable than one large model.

9. **Benchmark before you ship**: Create a simple 10-task benchmark. Run it against each model before recommending it.

10. **The best small model today is Qwen2.5 7B**: For agentic tool calling with Ollama, it has the best accuracy-to-size ratio as of March 2026. Gemini 2.0 Flash is the best free cloud fallback.
