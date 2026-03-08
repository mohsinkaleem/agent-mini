"""Core agent loop — think → act → observe → repeat."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from ..providers.base import BaseProvider
from .context import build_system_prompt
from .memory import Memory
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
        self.max_iterations: int = config.get("agent", {}).get("maxIterations", 30)
        self.temperature: float = config.get("agent", {}).get("temperature", 0.7)
        # Token/cost tracking per session
        self.session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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

            response = await self._call_provider_with_retry(
                messages, tool_defs, on_stream
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
                # Summarize history if too long
                if len(conversation) > 50:
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

        return "Reached maximum iterations. Please try a simpler request."

    async def _summarize_history(self, conversation: list[dict]) -> None:
        """Summarize oldest messages in-place to keep context bounded."""
        # Take the oldest 20 messages, keep the rest
        to_summarize = conversation[:20]
        keep = conversation[20:]

        # Build a summarization request
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation segment concisely. "
                    "Preserve key decisions, file paths, code changes, technical "
                    "context, and any user preferences. Be factual and brief."
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
            # On failure, keep conversation intact rather than losing data
            return

        # Atomic replacement: build new list first, then swap
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
