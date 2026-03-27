"""Core agent loop — think → act → observe → repeat."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from ..providers.base import BaseProvider
from .context import build_system_prompt
from .memory import Memory
from .token_estimator import (
    classify_model_tier,
    estimate_messages_tokens,
    get_effective_context,
    get_output_limit,
    get_tier_max_iterations,
)
from .tools import ToolExecutor
from .vision import build_image_content_parts

log = logging.getLogger("agent-mini")
StreamEmitter = Callable[[str], Awaitable[None]]

# HTTP status codes considered transient (worth retrying)
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


@dataclass
class ToolEvent:
    """Emitted when a tool is called or produces a result."""
    name: str
    arguments: dict | None = None
    result_preview: str | None = None
    is_error: bool = False


ToolEventCallback = Callable[[ToolEvent], Awaitable[None]]


class AgentLoop:
    """Runs the agentic ReAct loop.

    1. Build messages (system + history + user message).
    2. Call the LLM.
    3. If the LLM returns tool calls → execute them, append results, goto 2.
    4. If the LLM returns text → return it to the caller.
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: dict,
        memory: Memory,
    ):
        self.provider = provider
        self.config = config
        self.memory = memory
        self.tools = ToolExecutor(config, memory)
        self._model_name: str = provider.model_name
        self._model_tier: str = classify_model_tier(self._model_name)
        self._effective_ctx: int = get_effective_context(self._model_name)
        self._output_limit: int = get_output_limit(self._model_name)
        # Max iterations: respect explicit user config, otherwise use tier default
        user_iters = config.get("agent", {}).get("maxIterations")
        self.max_iterations: int = (
            user_iters if user_iters is not None
            else get_tier_max_iterations(self._model_name)
        )
        self.temperature: float = config.get("agent", {}).get("temperature", 0.7)
        # Token/cost tracking per session
        self.session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Loop detection state
        self._recent_calls: list[str] = []

    async def close(self) -> None:
        """Clean up provider and tool resources."""
        await self.provider.close()
        await self.tools.close()

    async def run(
        self,
        user_message: str,
        conversation: list[dict],
        on_stream: StreamEmitter | None = None,
        on_tool_event: ToolEventCallback | None = None,
    ) -> str:
        """Process *user_message* and return the assistant's final text reply.

        *conversation* is mutated in-place (appended with the new user/assistant
        turns) so the caller can maintain session state.
        When *on_stream* is provided, partial text deltas are emitted in real time.
        """
        system_prompt = build_system_prompt(self.config, self.memory)
        tool_defs = self.tools.get_tool_defs()

        # Reset per-turn usage tracking
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Build the full message list for this request
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        # Build user message — detect image references for vision support
        image_parts = build_image_content_parts(user_message)
        if image_parts:
            messages.append({"role": "user", "content": image_parts})
        else:
            messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            log.debug("iteration %d / %d", iteration + 1, self.max_iterations)

            # Prune old tool results in-memory before sending to provider
            pruned = self._prune_tool_results(messages)

            response = await self._call_provider_with_retry(
                pruned, tool_defs, on_stream
            )
            if isinstance(response, str):
                # Error string from retry exhaustion
                return response

            # Accumulate token usage if available
            if hasattr(response, "usage") and response.usage:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = response.usage.get(key, 0)
                    self.turn_usage[key] += val
                    self.session_usage[key] += val

            # ----- text-only response → done -----
            if not response.tool_calls:
                final = response.content or "(empty response)"
                # Persist to conversation history
                conversation.append({"role": "user", "content": user_message})
                conversation.append({"role": "assistant", "content": final})
                # Token-aware compaction: summarize when conversation
                # exceeds 75% of the model's effective context budget.
                current_tokens = estimate_messages_tokens(conversation)
                if current_tokens > int(self._effective_ctx * 0.75):
                    await self._summarize_history(conversation)
                return final

            # ----- tool calls → execute in parallel and continue -----
            assistant_msg: dict = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments
                            ),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute all tool calls concurrently
            async def _exec(tc):
                log.info(
                    "🔧 %s(%s)",
                    tc.name,
                    json.dumps(tc.arguments, ensure_ascii=False)[:200],
                )
                if on_tool_event:
                    await on_tool_event(ToolEvent(name=tc.name, arguments=tc.arguments))
                result = await self.tools.execute(tc.name, tc.arguments)
                log.debug("   → %s", result[:300])
                if on_tool_event:
                    preview = result[:200]
                    await on_tool_event(ToolEvent(
                        name=tc.name,
                        result_preview=preview,
                        is_error=result.startswith("Error:"),
                    ))
                return tc, result

            results = await asyncio.gather(
                *[_exec(tc) for tc in response.tool_calls]
            )

            for tc, result in results:
                # Compress tool output based on model tier
                if len(result) > self._output_limit:
                    head = self._output_limit * 2 // 3
                    tail = self._output_limit // 3
                    result = (
                        result[:head]
                        + f"\n\n[... truncated {len(result) - head - tail} chars ...]\n\n"
                        + result[-tail:]
                    )
                # Self-reflection on errors: nudge the LLM to reason about failures
                if result.startswith("Error:"):
                    result = (
                        f"{result}\n\n"
                        "[The tool call failed. Analyze what went wrong "
                        "and try a different approach.]"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )

            # Loop detection: identical consecutive tool calls → nudge
            for tc, _result in results:
                sig = hashlib.md5(
                    json.dumps({"n": tc.name, "a": tc.arguments}, sort_keys=True).encode()
                ).hexdigest()
                self._recent_calls.append(sig)
            # Keep a sliding window of last few signatures
            self._recent_calls = self._recent_calls[-6:]
            if len(self._recent_calls) >= 4:
                last4 = self._recent_calls[-4:]
                if last4[0] == last4[1] == last4[2] == last4[3]:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have repeated the same tool call multiple times "
                            "with the same result. Try a completely different approach."
                        ),
                    })

        return "Reached maximum iterations. Please try a simpler request."

    def _prune_tool_results(self, messages: list[dict]) -> list[dict]:
        """Return a copy of *messages* with old tool results trimmed.

        Protects the last 3 assistant turns and their tool results.
        Older tool messages get soft-trimmed (head+tail) or replaced
        with a placeholder.  The original list is never mutated.
        """
        # Find positions of the last 3 assistant messages
        asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        cutoff = asst_indices[-3] if len(asst_indices) >= 3 else 0

        pruned: list[dict] = []
        for i, msg in enumerate(messages):
            if i >= cutoff or msg.get("role") != "tool":
                pruned.append(msg)
                continue
            content = msg.get("content", "")
            if len(content) <= 4000:
                pruned.append(msg)
            elif i >= cutoff - 6:
                # Soft-trim: keep head + tail
                trimmed = content[:1500] + "\n...\n" + content[-1500:]
                pruned.append({**msg, "content": trimmed})
            else:
                # Hard-clear: replace with placeholder
                pruned.append({**msg, "content": "[Old tool result cleared to save context]"})
        return pruned

    async def _summarize_history(self, conversation: list[dict]) -> None:
        """Summarize oldest messages in-place to keep context bounded."""
        # Determine how many messages to summarize: enough to drop below
        # 50% of effective context, but at least 6 messages.
        target = int(self._effective_ctx * 0.5)
        tokens_so_far = 0
        cut = 0
        for i, msg in enumerate(conversation):
            content = msg.get("content", "")
            tokens_so_far += len(content) // 4 + 4 if isinstance(content, str) else 50
            if tokens_so_far > (estimate_messages_tokens(conversation) - target):
                cut = max(i, 6)
                break
        if cut < 6:
            cut = min(20, len(conversation) - 4)  # fallback: oldest 20, keep last 4

        to_summarize = conversation[:cut]
        keep = conversation[cut:]

        # Build a tight summarization request
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize in under 200 words. Keep file paths, error "
                    "messages, code snippets, and key decisions verbatim. "
                    "State what was done and what remains."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    f"[{m['role']}]: {m.get('content', '')[:500]}"
                    for m in to_summarize
                    if m.get("content")
                ),
            },
        ]

        try:
            response = await self.provider.chat(
                summary_messages, tools=None, temperature=0.3
            )
            summary_text = response.content or "Previous conversation context."
        except Exception as e:
            log.warning("Failed to summarize history: %s", e)
            return

        new_conversation = [
            {
                "role": "system",
                "content": f"[Previous context summary]\n{summary_text}",
            },
        ]
        new_conversation.extend(keep)
        conversation.clear()
        conversation.extend(new_conversation)

    async def _call_provider_with_retry(self, messages, tool_defs, on_stream):
        """Call the provider with retry + exponential backoff (with jitter) for transient errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if on_stream:
                    return await self.provider.chat_stream(
                        messages,
                        on_delta=on_stream,
                        tools=tool_defs or None,
                        temperature=self.temperature,
                    )
                else:
                    return await self.provider.chat(
                        messages,
                        tools=tool_defs or None,
                        temperature=self.temperature,
                    )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in _TRANSIENT_CODES and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)  # jitter
                    log.warning(
                        "Transient HTTP %d, retrying in %.1fs (attempt %d/%d)",
                        e.response.status_code, wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                log.error("Provider error: %s", e)
                return f"Error communicating with LLM: {e}"
            except (httpx.ConnectError, httpx.ReadTimeout, OSError) as e:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)  # jitter
                    log.warning(
                        "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                        wait, attempt + 1, max_retries, e,
                    )
                    await asyncio.sleep(wait)
                    continue
                log.error("Provider error after %d attempts: %s", max_retries, e)
                return f"Error communicating with LLM: {e}"
            except Exception as e:
                log.error("Provider error: %s", e)
                return f"Error communicating with LLM: {e}"
