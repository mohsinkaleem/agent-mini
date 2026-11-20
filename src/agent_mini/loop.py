"""The agent loop: ask the model, run the tools it calls, repeat until it answers.

This is the heart of the agent, and ``AgentLoop.run()`` is the place to start
reading. Stripped of everything else, it is:

    messages = [system prompt, *history, user message]
    repeat up to max_iterations times:
        reply = model(messages, tool schemas)
        if the reply has no tool calls: it is the answer; save the turn and return it
        run the tool calls and append their results to messages

The rest of this file handles what goes wrong in practice: a provider that
fails (retries), a model stuck repeating itself (loop detection) and running
out of iterations (a final summary). Fitting the messages into the context
budget is in ``context.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

import httpx

from .config import with_defaults
from .context import SUMMARIZE_AT, clip, prune, summarize_history
from .memory import Memory
from .prompt import build_system_prompt, context_overhead, estimate_messages_tokens, get_profile, num_ctx_for
from .providers.base import ChatResponse, ModelInfo, Provider, StreamCallback, ToolCall
from .tools import Tools

log = logging.getLogger("agent-mini")

# Worth retrying: rate limits, server hiccups, dropped connections. Read
# timeouts are not: a model that took 300 s once will do it again.
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIES = 3

# Messages the loop adds when the model needs a push.
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

# The trace saved with each reply ("[tools used: read_file(notes.txt), …]"):
# the argument shown for each tool, and how many calls are listed.
_TRACE_KEYS = ("path", "command", "query", "pattern", "url", "key")
_TRACE_MAX = 12
# Loop detection compares calls by name, arguments and the start of the result.
_SIGNATURE_RESULT_CHARS = 500


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


@dataclass
class _Turn:
    """What one ``run()`` keeps track of.

    It lives only as long as the run, so two runs on the same agent at once
    can't mix up each other's state.
    """

    user_message: str
    conversation: list[dict]  # the saved history; the finished turn is appended to it
    messages: list[dict]  # what the model sees: system prompt, history, this turn so far
    tool_defs: list[dict]
    trace: list[str] = field(default_factory=list)  # e.g. "read_file(notes.txt)"
    signatures: list[str] = field(default_factory=list)  # one per tool call, for loop detection
    tool_log: list[dict] = field(default_factory=list)  # for the run report
    loop_nudges: int = 0
    nudged_empty: bool = False


def _trace_entry(name: str, arguments: dict) -> str:
    """A short form of a call for the trace, e.g. ``read_file(notes.txt)``."""
    for key in _TRACE_KEYS:
        if key in arguments:
            value = " ".join(str(arguments[key]).split())  # on one line
            if len(value) > 60:
                value = value[:57] + "…"
            return f"{name}({value})"
    return name


def _is_looping(signatures: list[str]) -> bool:
    """Four identical calls in a row (A A A A), or two alternating ones (A B A B)."""
    if len(signatures) < 4:
        return False
    a, b, c, d = signatures[-4:]
    return a == c and b == d


def _with_note(text: str, note: str) -> str:
    return f"{text}\n\n{note}" if text else note


class AgentLoop:
    """Runs the loop for one model, its tools and memory."""

    def __init__(self, provider: Provider, config: dict, memory: Memory):
        self.provider = provider
        self.config = with_defaults(config)
        self.memory = memory
        self.tools = Tools(self.config, memory)
        self.model_info: ModelInfo | None = None
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
            # The model's window is smaller than the tier's budget needs: shrink the budget to fit.
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
        turn = self._start_turn(user_message, conversation)

        for iteration in range(1, self.max_iterations + 1):
            self.turn_iterations = iteration

            # 1. Ask the model, sending only what fits the context budget.
            try:
                response = await self._call(self._prune(turn.messages), turn.tool_defs, on_stream, on_thinking)
            except ProviderError as e:
                return self._finish(turn, "provider_error", f"Error communicating with LLM: {e}",
                                    saved="[No reply: the model provider failed.]")
            self._add_usage(response.usage)
            calls, text = response.tool_calls or [], (response.content or "").strip()

            # 2. No tool calls: the model has answered.
            if not calls:
                if not text and turn.tool_log and not turn.nudged_empty:
                    # Small-model fix: an empty reply after using tools gets one nudge.
                    turn.nudged_empty = True
                    turn.messages.append({"role": "user", "content": _EMPTY_NUDGE})
                    continue
                reply = self._finish(turn, "text", text or "(empty response)")
                if estimate_messages_tokens(conversation) > self.profile.context * SUMMARIZE_AT:
                    await self._summarize_history(conversation)
                return reply

            # 3. Run the tools and append their results; the next iteration shows them to the model.
            await self._run_tools(turn, calls, text, on_tool_event)

            # 4. Same calls with the same results: nudge once; the second time, stop and ask for an answer.
            if _is_looping(turn.signatures):
                turn.signatures.clear()
                turn.loop_nudges += 1
                if turn.loop_nudges >= 2:
                    log.warning("Agent is repeating itself; asking for a final answer")
                    reply = await self._final_answer(turn.messages, _STUCK_PROMPT)
                    return self._finish(turn, "stuck", _with_note(reply, "[Stopped: repeated the same tool calls.]"))
                log.info("Loop detected: nudging the model to change approach")
                turn.messages.append({"role": "user", "content": _LOOP_NUDGE})

        # Out of iterations: one last call without tools, so the user gets a summary instead of nothing.
        reply = await self._final_answer(turn.messages, _LIMIT_PROMPT)
        note = f"[Stopped: reached max iterations ({self.max_iterations}) before finishing.]"
        return self._finish(turn, "max_iterations", _with_note(reply, note))

    # ── The steps of a turn ──────────────────────────────────────────

    def _start_turn(self, user_message: str, conversation: list[dict]) -> _Turn:
        """Build the messages for this turn and reset the per-turn counters."""
        names = self.tools.names()
        memories = self.memory.recent(self.profile.memory_items) if "recall" in names else []
        system = build_system_prompt(self.config, names, self.profile, memories)
        turn = _Turn(
            user_message=user_message,
            conversation=conversation,
            messages=[{"role": "system", "content": system}, *conversation, {"role": "user", "content": user_message}],
            tool_defs=self.tools.definitions(),
        )
        self.turn_usage = {k: 0 for k in self.session_usage}
        # Kept up to date as the turn goes, so an interrupted run can still be reported.
        self.last_run = {"finish_reason": "interrupted", "reply": "", "tool_calls": turn.tool_log,
                         "usage": self.turn_usage, "messages": turn.messages}
        return turn

    def _add_usage(self, usage: dict | None) -> None:
        for key, value in (usage or {}).items():
            if key in self.turn_usage:
                self.turn_usage[key] += value or 0
                self.session_usage[key] += value or 0

    async def _run_tools(self, turn: _Turn, calls: list[ToolCall], text: str,
                         on_tool_event: ToolEventCallback | None) -> None:
        """Run the calls in order; append the assistant's message and every result to the messages."""
        turn.messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in calls
            ],
        })
        for tc in calls:
            # One at a time, not in parallel: models expect a later call to see what an
            # earlier one did ("write a file" then "run it" in the same reply).
            result = await self._run_tool(tc, on_tool_event)
            turn.trace.append(_trace_entry(tc.name, tc.arguments))
            turn.tool_log.append({"name": tc.name, "args": tc.arguments, "error": result.startswith("Error:")})
            signature = [tc.name, tc.arguments, result[:_SIGNATURE_RESULT_CHARS]]
            turn.signatures.append(json.dumps(signature, sort_keys=True, default=str))
            # The tool_call_id ties each result to the call that asked for it.
            turn.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": clip(tc.name, result, self.profile.output_limit)})

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

    def _finish(self, turn: _Turn, reason: str, reply: str, saved: str | None = None) -> str:
        """End the turn: save it to the conversation, record how it ended, return the reply.

        *saved*, if given, is what the history keeps instead of the reply.
        """
        self._record(turn.conversation, turn.user_message, saved or reply, turn.trace)
        self.finish_reason = reason
        self.last_run.update(finish_reason=reason, reply=reply, iterations=self.turn_iterations,
                             loop_nudges=turn.loop_nudges)
        return reply

    @staticmethod
    def _record(conversation: list[dict], user_message: str, reply: str, trace: list[str]) -> None:
        """Save the turn. The tool trace tells later turns which files were touched."""
        if trace:
            shown = trace[:_TRACE_MAX] + (["…"] if len(trace) > _TRACE_MAX else [])
            reply += "\n\n[tools used: " + ", ".join(shown) + "]"
        conversation += [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]

    # ── Talking to the model ─────────────────────────────────────────

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

    # ── Context budget (see context.py) ──────────────────────────────

    def _prune(self, messages: list[dict]) -> list[dict]:
        return prune(messages, self.profile)

    async def _summarize_history(self, conversation: list[dict]) -> None:
        await summarize_history(self.provider, conversation, self.profile.context)
