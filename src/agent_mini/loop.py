"""The agent loop: ask the model, run the tools it calls, repeat until it answers."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

import httpx

from .config import with_defaults
from .memory import Memory
from .prompt import build_system_prompt, context_overhead, estimate_messages_tokens, get_profile, num_ctx_for
from .providers.base import ChatResponse, ModelInfo, Provider, StreamCallback, ToolCall, parse_text_tool_calls
from .tools import Tools
from .vision import image_parts

log = logging.getLogger("agent-mini")

# Worth retrying: rate limits, server hiccups, dropped connections. Read
# timeouts are not: a model that took 300 s once will do it again.
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIES = 3

_SUMMARY_MARKER = "[Summary of the earlier conversation]"
_LOOP_NUDGE = "You repeated the same tool calls and got the same results. Try a different approach."
_EMPTY_NUDGE = "Your last reply was empty. Reply to the user with a short answer."
_STUCK_PROMPT = (
    "You are repeating the same tool calls without making progress. Do not call any more tools. "
    "Tell the user what you tried, what went wrong, and what they could do next."
)
_LIMIT_PROMPT = (
    "You have reached the tool-call limit for this request. Do not call any more tools. "
    "Summarize what you did, what is left to do, and any partial result."
)
# The argument shown for each tool in the trace saved with a reply.
_TRACE_KEYS = ("path", "command", "query", "pattern", "url", "key")


class ProviderError(Exception):
    """The model server could not be reached or kept failing."""


@dataclass
class ToolEvent:
    """A tool starting (``arguments`` set) or finishing (``result`` set)."""

    name: str
    arguments: dict | None = None
    result: str | None = None
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
    return len(signatures) >= 4 and signatures[-4] == signatures[-2] and signatures[-3] == signatures[-1]


def _with_note(text: str, note: str) -> str:
    return f"{text}\n\n{note}" if text else note


class AgentLoop:
    """Runs the ReAct loop for one model, its tools and memory.

    1. Send system prompt + history + user message.
    2. If the model calls tools, run them, append the results, go to 1.
    3. If it answers in text, save the turn and return the answer.
    """

    def __init__(self, provider: Provider, config: dict, memory: Memory):
        self.provider = provider
        self.config = with_defaults(config)
        self.memory = memory
        self.tools = Tools(self.config, memory)
        self.model_info: ModelInfo | None = None
        # The gateway turns this off: remote users must not attach local files.
        self.allow_local_images = True
        self.session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.turn_usage = dict(self.session_usage)
        self.turn_iterations = 0
        # How the last run() ended: "text", "max_iterations", "stuck" or "provider_error".
        self.finish_reason = ""
        # What the last run() did, for `chat --report` and the eval harness.
        self.last_run: dict = {}
        self._apply_profile()

    def _apply_profile(self) -> None:
        """Budgets from the tier, the config and what the server reported."""
        agent_cfg = self.config["agent"]
        info = self.model_info
        profile = get_profile(self.provider.model, agent_cfg, params_b=info.params_b if info else None)
        defs = self.tools.definitions()
        num_ctx = num_ctx_for(profile, defs)
        if info and info.context_length:
            overhead = context_overhead(defs)
            if profile.context + overhead > info.context_length:
                profile = replace(profile, context=max(1024, info.context_length - overhead))
            num_ctx = min(num_ctx, info.context_length)
        self.profile = profile
        self.max_iterations: int = agent_cfg["maxIterations"] or profile.max_iterations
        self.temperature: float = profile.temperature if agent_cfg["temperature"] is None else agent_cfg["temperature"]
        self.provider.context_window = num_ctx

    async def set_provider(self, provider: Provider) -> None:
        """Switch model (``/model``) and recompute every tier-derived budget."""
        old, self.provider, self.model_info = self.provider, provider, None
        self._apply_profile()
        await old.close()

    async def detect_model(self) -> ModelInfo | None:
        """Ask the server about the model and re-derive the budgets from the real numbers."""
        if info := await self.provider.model_info():
            self.model_info = info
            self._apply_profile()
        return info

    async def close(self) -> None:
        await self.provider.close()
        await self.tools.close()

    # ── The loop ─────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        conversation: list[dict],
        on_stream: StreamCallback | None = None,
        on_tool_event: ToolEventCallback | None = None,
        on_thinking: StreamCallback | None = None,
    ) -> str:
        """Answer *user_message*. The turn is appended to *conversation* in place."""
        names = self.tools.names()
        memories = self.memory.recent(self.profile.memory_items) if "recall" in names else []
        system = build_system_prompt(self.config, names, self.profile, memories)
        content = image_parts(user_message, self.tools.workspace, self.allow_local_images) or user_message
        messages = [{"role": "system", "content": system}, *conversation, {"role": "user", "content": content}]
        defs, known = self.tools.definitions(), self.tools.known_names()

        # Per-run state is local, so concurrent gateway users can't mix it up.
        self.turn_usage = {k: 0 for k in self.session_usage}
        trace: list[str] = []
        signatures: list[str] = []
        tool_log: list[dict] = []
        nudges = text_calls = 0
        nudged_empty = False

        # Kept up to date, so an interrupted run can still be reported.
        self.last_run = {"finish_reason": "interrupted", "reply": "", "tool_calls": tool_log,
                         "usage": self.turn_usage, "messages": messages}

        def finish(reason: str, reply: str, saved: str | None = None) -> str:
            self._record(conversation, user_message, saved or reply, trace)
            self.finish_reason = reason
            self.last_run.update(finish_reason=reason, reply=reply, iterations=self.turn_iterations,
                                 text_tool_calls=text_calls, loop_nudges=nudges)
            return reply

        for iteration in range(1, self.max_iterations + 1):
            self.turn_iterations = iteration
            try:
                response = await self._call(self._prune(messages), defs, on_stream, on_thinking)
            except ProviderError as e:
                return finish("provider_error", f"Error communicating with LLM: {e}",
                              "[No reply: the model provider failed.]")
            for key, value in (response.usage or {}).items():
                if key in self.turn_usage:
                    self.turn_usage[key] += value or 0
                    self.session_usage[key] += value or 0

            calls, text = response.tool_calls or [], (response.content or "").strip()
            if not calls and text:
                calls, text = parse_text_tool_calls(text, known)
                if calls:
                    text_calls += len(calls)
                    log.info("Recovered %d tool call(s) written as text", len(calls))
            if not calls:
                if not text and tool_log and not nudged_empty:
                    nudged_empty = True
                    messages.append({"role": "user", "content": _EMPTY_NUDGE})
                    continue
                reply = finish("text", text or "(empty response)")
                if estimate_messages_tokens(conversation) > self.profile.context * 0.75:
                    await self._summarize_history(conversation)
                return reply

            messages.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in calls
            ]})
            results = await asyncio.gather(*(self._run_tool(tc, on_tool_event) for tc in calls))
            for tc, result in zip(calls, results):
                trace.append(_trace_entry(tc.name, tc.arguments))
                tool_log.append({"name": tc.name, "args": tc.arguments, "error": result.startswith("Error:")})
                signatures.append(json.dumps([tc.name, tc.arguments, result[:500]], sort_keys=True, default=str))
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                 "content": self._clip(tc.name, result)})

            # Same calls with the same results: nudge once, then stop and ask for an answer.
            if _is_looping(signatures):
                signatures.clear()
                nudges += 1
                if nudges >= 2:
                    log.warning("Agent is repeating itself; asking for a final answer")
                    reply = await self._final_answer(messages, _STUCK_PROMPT)
                    return finish("stuck", _with_note(reply, "[Stopped: repeated the same tool calls.]"))
                log.info("Loop detected: nudging the model to change approach")
                messages.append({"role": "user", "content": _LOOP_NUDGE})

        # Out of iterations: one last call without tools, so the user gets a summary instead of nothing.
        reply = await self._final_answer(messages, _LIMIT_PROMPT)
        note = f"[Stopped: reached max iterations ({self.max_iterations}) before finishing.]"
        return finish("max_iterations", _with_note(reply, note))

    @staticmethod
    def _record(conversation: list[dict], user_message: str, reply: str, trace: list[str]) -> None:
        """Save the turn. The tool trace tells later turns which files were touched."""
        if trace:
            reply += "\n\n[tools used: " + ", ".join(trace[:12] + (["…"] if len(trace) > 12 else [])) + "]"
        conversation += [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]

    async def _run_tool(self, tc: ToolCall, on_tool_event: ToolEventCallback | None) -> str:
        if on_tool_event:
            await on_tool_event(ToolEvent(tc.name, arguments=tc.arguments))
        started = time.monotonic()
        result = await self.tools.execute(tc.name, tc.arguments)
        log.debug("%s(%s) → %s", tc.name, json.dumps(tc.arguments, ensure_ascii=False)[:200], result[:300])
        if on_tool_event:
            await on_tool_event(ToolEvent(tc.name, result=result, is_error=result.startswith("Error:"),
                                          duration=time.monotonic() - started))
        return result

    def _clip(self, name: str, result: str) -> str:
        """Head and tail of a result longer than the tier allows."""
        limit = self.profile.output_limit
        if len(result) <= limit:
            return result
        head, tail = limit * 2 // 3, limit // 3
        hint = {"read_file": " Use offset/limit to read a specific range.",
                "shell": " Pipe the output through grep, head or tail to see the part you need."}.get(name, "")
        return f"{result[:head]}\n\n[... {len(result) - head - tail} characters cut.{hint}]\n\n{result[-tail:]}"

    async def _call(self, messages: list[dict], defs: list[dict], on_stream, on_thinking) -> ChatResponse:
        """Call the model, retrying transient failures with jittered backoff."""
        streamed = False

        async def emit(delta: str) -> None:
            nonlocal streamed
            streamed = True
            await on_stream(delta)

        for attempt in range(_RETRIES):
            try:
                return await self.provider.chat(messages, tools=defs or None, temperature=self.temperature,
                                                on_delta=emit if on_stream else None, on_thinking=on_thinking)
            except Exception as e:
                transient = isinstance(e, _TRANSIENT_ERRORS) or (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code in _TRANSIENT_CODES
                )
                # A retry after text was shown would show it twice.
                if transient and not streamed and attempt < _RETRIES - 1:
                    wait = 2**attempt + random.uniform(0, 1)
                    log.warning("Transient error, retry %d/%d in %.1fs: %s", attempt + 1, _RETRIES, wait, e)
                    await asyncio.sleep(wait)
                    continue
                log.error("Provider error: %s", e)
                raise ProviderError(str(e)) from e
        raise AssertionError("unreachable")

    async def _final_answer(self, messages: list[dict], instruction: str) -> str:
        """A plain-text answer with tools disabled; '' if the call fails."""
        request = self._prune([*messages, {"role": "user", "content": instruction}])
        try:
            response = await self.provider.chat(request, tools=None, temperature=self.temperature)
        except Exception as e:
            log.warning("Final answer call failed: %s", e)
            return ""
        return (response.content or "").strip()

    # ── Context budget ───────────────────────────────────────────────

    def _prune(self, messages: list[dict]) -> list[dict]:
        """A copy of *messages* that fits the context budget.

        Tool results older than the last three assistant turns are cut to head
        and tail. If the request is still too big, the oldest results (never
        the latest round) become one-line stubs.
        """
        asst = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        cutoff = asst[-3] if len(asst) >= 3 else 0
        soft = max(1000, self.profile.output_limit // 2)
        pruned = []
        for i, msg in enumerate(messages):
            text = msg.get("content") or ""
            if i < cutoff and msg["role"] == "tool" and len(text) > soft:
                msg = {**msg, "content": text[: soft // 2] + "\n...\n" + text[-soft // 2:]}
            pruned.append(msg)

        total = estimate_messages_tokens(pruned)
        for i in range(asst[-1] if asst else 0):
            if total <= self.profile.context:
                break
            msg = pruned[i]
            if msg["role"] != "tool" or msg["content"].startswith("[cleared"):
                continue
            stub = {**msg, "content": f"[cleared to save context: {msg.get('name', 'tool')} output, "
                                      f"{len(msg['content'])} chars. Run the tool again if you need it.]"}
            total -= estimate_messages_tokens([msg]) - estimate_messages_tokens([stub])
            pruned[i] = stub
        return pruned

    async def _summarize_history(self, conversation: list[dict]) -> None:
        """Replace older turns with a short summary so the history stays within budget.

        The latest exchange is always kept, and the kept part starts on a user turn.
        """
        target = self.profile.context // 2
        cut = len(conversation) - 2
        while cut >= 2 and estimate_messages_tokens(conversation[cut - 2:]) <= target:
            cut -= 2
        while cut > 0 and conversation[cut]["role"] != "user":
            cut -= 1
        if cut <= 0:
            return
        lines = []
        for m in conversation[:cut]:
            text = m.get("content")
            if isinstance(text, str) and text:
                # An earlier summary is carried over whole; other messages are clipped.
                lines.append(f"[{m['role']}]: {text if text.startswith(_SUMMARY_MARKER) else text[:500]}")
        request = [
            {"role": "system", "content": "Summarize in under 200 words. Keep file paths, error messages, code "
                                          "snippets and key decisions verbatim. Say what was done and what remains."},
            {"role": "user", "content": "\n".join(lines)},
        ]
        try:
            response = await self.provider.chat(request, tools=None, temperature=0.3)
        except Exception as e:
            log.warning("Failed to summarize history: %s", e)
            return
        # A user message, not a second system message: small models anchor on one system prompt.
        summary = {"role": "user", "content": f"{_SUMMARY_MARKER}\n{response.content or 'Earlier conversation.'}"}
        conversation[:] = [summary, *conversation[cut:]]
