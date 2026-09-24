"""Core agent loop — think → act → observe → repeat."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

import httpx

from ..providers.base import BaseProvider, ModelInfo, StreamCallback, parse_text_tool_calls
from .context import build_system_prompt
from .memory import Memory
from .token_estimator import (
    context_overhead,
    estimate_messages_tokens,
    get_profile,
    num_ctx_for,
)
from .tools import ToolExecutor
from .vision import build_image_content_parts

log = logging.getLogger("agent-mini")

# HTTP status codes considered transient (worth retrying)
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
# Connection failures worth retrying. Read timeouts are not: a model that
# took 300 s once will do it again.
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)

_SUMMARY_MARKER = "[Previous context summary]"
_LOOP_NUDGE = (
    "You have repeated the same tool call multiple times with the same result. "
    "Try a completely different approach."
)
_STUCK_PROMPT = (
    "You are repeating the same tool calls without making progress. Do not call "
    "any more tools. Tell the user what you tried, what went wrong, and what they "
    "could do next."
)
_LIMIT_PROMPT = (
    "You have reached the tool-call limit for this request. Do not call any more "
    "tools. Summarize what you did, what is left to do, and any partial result."
)
# Argument shown for each tool in the per-turn trace.
_TRACE_KEYS = ("path", "command", "query", "pattern", "url", "key")


@dataclass
class ToolEvent:
    """Emitted when a tool is called or produces a result."""
    name: str
    arguments: dict | None = None
    result_preview: str | None = None
    is_error: bool = False
    duration: float = 0.0


ToolEventCallback = Callable[[ToolEvent], Awaitable[None]]


def _trace_entry(name: str, arguments: dict) -> str:
    for key in _TRACE_KEYS:
        if key in arguments:
            value = " ".join(str(arguments[key]).split())
            return f"{name}({value[:57] + '…' if len(value) > 60 else value})"
    return name


def _is_looping(signatures: list[str]) -> bool:
    """Four identical calls in a row, or an A→B→A→B oscillation."""
    if len(signatures) < 4:
        return False
    a, b, c, d = signatures[-4:]
    return a == c and b == d


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
        agent_cfg = config.get("agent", {})
        # null in config means "let the provider pick" (reasoning models reject it).
        self.temperature: float | None = agent_cfg.get("temperature", 0.7)
        # Token/cost tracking per session
        self.session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_iterations = 0
        # How the last run() ended: "text", "max_iterations", "stuck" or "provider_error".
        self.finish_reason: str = ""
        # Tool calls recovered from plain text this session (see parse_text_tool_calls).
        self.text_tool_calls = 0
        # The gateway turns this off: remote users must not attach local files.
        self.allow_local_images = True
        self.model_info: ModelInfo | None = None
        self._apply_profile()

    def _apply_profile(self) -> None:
        """Derive tier budgets from the model name, config overrides and detected facts."""
        agent_cfg = self.config.get("agent", {})
        info = self.model_info
        profile = get_profile(self.provider.model_name, agent_cfg, params_b=info.params_b if info else None)
        tool_defs = self.tools.get_tool_defs()
        if info and info.context_length:
            overhead = context_overhead(tool_defs)
            if profile.context + overhead > info.context_length:
                profile = replace(profile, context=max(1024, info.context_length - overhead))
            num_ctx = min(num_ctx_for(profile, tool_defs), info.context_length)
        else:
            num_ctx = num_ctx_for(profile, tool_defs)
        self.profile = profile
        # Max iterations: respect explicit user config, otherwise use tier default
        user_iters = agent_cfg.get("maxIterations")
        self.max_iterations: int = (
            user_iters if user_iters is not None else profile.max_iterations
        )
        self.provider.context_window = num_ctx

    async def set_provider(self, provider: BaseProvider) -> None:
        """Switch model (``/model``) and recompute every tier-derived budget."""
        old = self.provider
        self.provider = provider
        self.model_info = None
        self._apply_profile()
        await old.close()

    async def detect_model(self) -> ModelInfo | None:
        """Ask the server about the model and re-derive budgets from the real numbers."""
        info = await self.provider.model_info()
        if info:
            self.model_info = info
            self._apply_profile()
        return info

    async def close(self) -> None:
        """Clean up provider and tool resources."""
        await self.provider.close()
        await self.tools.close()

    async def run(
        self,
        user_message: str,
        conversation: list[dict],
        on_stream: StreamCallback | None = None,
        on_tool_event: ToolEventCallback | None = None,
        on_thinking: StreamCallback | None = None,
    ) -> str:
        """Process *user_message* and return the assistant's final text reply.

        *conversation* is mutated in-place (appended with the new user/assistant
        turns) so the caller can maintain session state.
        When *on_stream* is provided, partial text deltas are emitted in real time
        (and thinking deltas through *on_thinking*).
        """
        tool_defs = self.tools.get_tool_defs()
        known_tools = {d["function"]["name"] for d in tool_defs}
        system_prompt = build_system_prompt(
            self.config,
            self.memory,
            model_name=self.provider.model_name,
            tool_defs=tool_defs,
            profile=self.profile,
        )

        # Reset per-turn usage tracking
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_iterations = 0

        # Build the full message list for this request
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        # Build user message — detect image references for vision support
        image_parts = build_image_content_parts(
            user_message, workspace=self.tools.workspace, allow_local=self.allow_local_images
        )
        if image_parts:
            messages.append({"role": "user", "content": image_parts})
        else:
            messages.append({"role": "user", "content": user_message})

        # Per-run state lives here, not on self, so concurrent gateway users can't mix it up.
        trace: list[str] = []
        signatures: list[str] = []
        nudges = 0

        for iteration in range(self.max_iterations):
            log.debug("iteration %d / %d", iteration + 1, self.max_iterations)
            self.turn_iterations = iteration + 1

            # Prune old tool results in-memory before sending to provider
            pruned = self._prune_tool_results(messages)

            response = await self._call_provider_with_retry(
                pruned, tool_defs, on_stream, on_thinking
            )
            if isinstance(response, str):
                # Error string from retry exhaustion. Record the turn so the
                # next one knows it happened (and what tools already ran).
                self._record(conversation, user_message, "[No reply: the model provider failed.]", trace)
                self.finish_reason = "provider_error"
                return response

            if response.usage:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = response.usage.get(key, 0)
                    self.turn_usage[key] += val
                    self.session_usage[key] += val

            calls = response.tool_calls
            content = response.content
            if not calls and content:
                calls, content = parse_text_tool_calls(content, known_tools)
                if calls:
                    self.text_tool_calls += len(calls)
                    log.info("Recovered %d tool call(s) written as text", len(calls))

            # ----- text-only response → done -----
            if not calls:
                final = content or "(empty response)"
                self._record(conversation, user_message, final, trace)
                # Token-aware compaction: summarize when conversation
                # exceeds 75% of the model's effective context budget.
                current_tokens = estimate_messages_tokens(conversation)
                if current_tokens > int(self.profile.context * 0.75):
                    await self._summarize_history(conversation)
                self.finish_reason = "text"
                return final

            # ----- tool calls → execute in parallel and continue -----
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in calls
                ],
            })

            results = await asyncio.gather(*[self._run_tool(tc, on_tool_event) for tc in calls])

            for tc, result in results:
                trace.append(_trace_entry(tc.name, tc.arguments))
                signatures.append(
                    json.dumps([tc.name, tc.arguments, result[:500]], sort_keys=True, default=str)
                )
                # Compress tool output based on model tier
                limit = self.profile.output_limit
                if len(result) > limit:
                    head = limit * 2 // 3
                    tail = limit // 3
                    hint = " Use offset/limit to read a specific range." if tc.name == "read_file" else ""
                    result = (
                        result[:head]
                        + f"\n\n[... truncated {len(result) - head - tail} chars ...{hint}]\n\n"
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

            # Loop detection: same calls with the same results → nudge once,
            # then stop and ask for an answer.
            if _is_looping(signatures):
                signatures.clear()
                nudges += 1
                if nudges >= 2:
                    log.warning("Agent is repeating itself; asking for a final answer")
                    final = await self._final_answer(messages, _STUCK_PROMPT)
                    final = (final + "\n\n" if final else "") + "[Stopped: repeated the same tool calls.]"
                    self._record(conversation, user_message, final, trace)
                    self.finish_reason = "stuck"
                    return final
                log.info("Loop detected: nudging the model to change approach")
                messages.append({"role": "user", "content": _LOOP_NUDGE})

        # Out of iterations: one last call without tools, so the user gets a
        # summary of the work instead of nothing.
        summary = await self._final_answer(messages, _LIMIT_PROMPT)
        final = (summary + "\n\n" if summary else "") + (
            f"[Stopped: reached max iterations ({self.max_iterations}) before finishing.]"
        )
        self._record(conversation, user_message, final, trace)
        self.finish_reason = "max_iterations"
        return final

    @staticmethod
    def _record(conversation: list[dict], user_message: str, reply: str, trace: list[str]) -> None:
        """Save the turn. The tool trace tells later turns which files were touched."""
        saved = reply
        if trace:
            shown = trace[:12] + (["…"] if len(trace) > 12 else [])
            saved += "\n\n[tools used: " + ", ".join(shown) + "]"
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": saved})

    async def _run_tool(self, tc, on_tool_event: ToolEventCallback | None):
        log.debug("tool call: %s(%s)", tc.name, json.dumps(tc.arguments, ensure_ascii=False)[:200])
        if on_tool_event:
            await on_tool_event(ToolEvent(name=tc.name, arguments=tc.arguments))
        started = time.monotonic()
        result = await self.tools.execute(tc.name, tc.arguments)
        elapsed = time.monotonic() - started
        log.debug("   → (%.2fs) %s", elapsed, result[:300])
        if on_tool_event:
            await on_tool_event(ToolEvent(
                name=tc.name,
                result_preview=result[:200],
                is_error=result.startswith("Error:"),
                duration=elapsed,
            ))
        return tc, result

    async def _final_answer(self, messages: list[dict], instruction: str) -> str:
        """Ask for a plain-text answer with tools disabled. Returns '' on failure."""
        request = self._prune_tool_results([*messages, {"role": "user", "content": instruction}])
        try:
            response = await self.provider.chat(request, tools=None, temperature=self.temperature)
        except Exception as e:
            log.warning("Final answer call failed: %s", e)
            return ""
        return (response.content or "").strip()

    def _prune_tool_results(self, messages: list[dict]) -> list[dict]:
        """Return a copy of *messages* that fits the context budget.

        Tool results older than the last 3 assistant turns are trimmed to
        head+tail. If the request is still over budget, the oldest results
        (never the latest round) are replaced with a one-line stub. The
        original list is never mutated.
        """
        asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        cutoff = asst_indices[-3] if len(asst_indices) >= 3 else 0
        soft = max(1000, self.profile.output_limit // 2)

        pruned: list[dict] = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if i < cutoff and msg.get("role") == "tool" and len(content) > soft:
                half = soft // 2
                msg = {**msg, "content": content[:half] + "\n...\n" + content[-half:]}
            pruned.append(msg)

        budget = self.profile.context
        total = estimate_messages_tokens(pruned)
        if total <= budget:
            return pruned
        latest = asst_indices[-1] if asst_indices else len(pruned)
        for i in range(latest):
            if total <= budget:
                break
            msg = pruned[i]
            if msg.get("role") != "tool" or msg.get("content", "").startswith("[cleared"):
                continue
            stub = {
                **msg,
                "content": (
                    f"[cleared to save context: {msg.get('name', 'tool')} output, "
                    f"{len(msg.get('content', ''))} chars. Re-run the tool if you need it.]"
                ),
            }
            total -= estimate_messages_tokens([msg]) - estimate_messages_tokens([stub])
            pruned[i] = stub
        return pruned

    async def _summarize_history(self, conversation: list[dict]) -> None:
        """Summarize oldest messages in-place to keep context bounded."""
        # Determine how many messages to summarize: enough to drop below
        # 50% of effective context, but at least 6 messages.
        total = estimate_messages_tokens(conversation)
        target = int(self.profile.context * 0.5)
        acc, cut = 0, 0
        for i, msg in enumerate(conversation):
            acc += estimate_messages_tokens([msg])
            if acc > total - target:
                cut = max(i, 6)
                break
        if cut < 6:
            cut = min(20, len(conversation) - 4)  # fallback: oldest 20, keep last 4
        # Always keep the latest exchange, and start the kept part on a user turn.
        cut = min(cut, len(conversation) - 2)
        while cut > 0 and conversation[cut].get("role") != "user":
            cut -= 1
        if cut <= 0:
            return

        to_summarize = conversation[:cut]
        keep = conversation[cut:]

        # An earlier summary is carried over whole; other messages are clipped.
        lines = []
        for m in to_summarize:
            text = m.get("content")
            if not isinstance(text, str) or not text:
                continue
            if not text.startswith(_SUMMARY_MARKER):
                text = text[:500]
            lines.append(f"[{m['role']}]: {text}")

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
            {"role": "user", "content": "\n".join(lines)},
        ]

        try:
            response = await self.provider.chat(
                summary_messages, tools=None, temperature=0.3
            )
            summary_text = response.content or "Previous conversation context."
        except Exception as e:
            log.warning("Failed to summarize history: %s", e)
            return

        # Use role='user' rather than 'system' — the real system prompt is
        # prepended fresh every turn in run(), so a second 'system' entry
        # here would give small models two leading system messages and
        # dilute the constraint-first anchoring they rely on.
        new_conversation = [
            {
                "role": "user",
                "content": f"{_SUMMARY_MARKER}\n{summary_text}",
            },
        ]
        new_conversation.extend(keep)
        conversation.clear()
        conversation.extend(new_conversation)

    async def _call_provider_with_retry(self, messages, tool_defs, on_stream, on_thinking=None):
        """Call the provider with retry + exponential backoff (with jitter) for transient errors."""
        max_retries = 3
        emitted = False

        async def _emit(delta: str) -> None:
            nonlocal emitted
            emitted = True
            await on_stream(delta)

        for attempt in range(max_retries):
            try:
                if on_stream:
                    return await self.provider.chat_stream(
                        messages,
                        on_delta=_emit,
                        tools=tool_defs or None,
                        temperature=self.temperature,
                        on_thinking=on_thinking,
                    )
                return await self.provider.chat(
                    messages,
                    tools=tool_defs or None,
                    temperature=self.temperature,
                )
            except Exception as e:
                transient = (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code in _TRANSIENT_CODES
                ) or isinstance(e, _TRANSIENT_ERRORS)
                # A retry after text was shown would show it twice.
                if transient and not emitted and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)  # jitter
                    log.warning(
                        "Transient error, retry %d/%d in %.1fs: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    await asyncio.sleep(wait)
                    continue
                log.error("Provider error: %s", e)
                return f"Error communicating with LLM: {e}"
