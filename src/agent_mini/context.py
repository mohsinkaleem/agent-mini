"""Keeping the conversation inside the model's context budget.

Small models lose accuracy long before their context window is full, so each
tier has a token budget (see ``prompt.py``). Three functions keep the messages
inside it:

- ``clip``: a single tool result longer than the tier allows keeps its head and tail.
- ``prune``: before every model call, old tool results are shortened, then
  replaced by one-line stubs while the request is still over budget.
- ``summarize_history``: after a turn, once the saved history is more than
  ``SUMMARIZE_AT`` of the budget, the oldest turns become one summary.
"""

from __future__ import annotations

import logging

from .prompt import TierProfile, estimate_messages_tokens
from .providers.base import Provider

log = logging.getLogger("agent-mini")

SUMMARY_MARKER = "[Summary of the earlier conversation]"
SUMMARIZE_AT = 0.75  # summarize the history once it fills this much of the budget
_KEEP_FULL_TURNS = 3  # tool results from the last three assistant turns are never shortened
_SUMMARY_INPUT_CHARS = 500  # of each message, sent to the summarizer

_CLIP_HINTS = {
    "read_file": " Use offset/limit to read a specific range.",
    "shell": " Pipe the output through grep, head or tail to see the part you need.",
}


def clip(name: str, result: str, limit: int) -> str:
    """The head (2/3) and tail (1/3) of a tool result longer than *limit* characters."""
    if len(result) <= limit:
        return result
    head, tail = limit * 2 // 3, limit // 3
    cut = len(result) - head - tail
    return f"{result[:head]}\n\n[... {cut} characters cut.{_CLIP_HINTS.get(name, '')}]\n\n{result[-tail:]}"


def prune(messages: list[dict], profile: TierProfile) -> list[dict]:
    """A copy of *messages* that fits the context budget. *messages* is not changed."""
    assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    # Step 1: tool results older than the last few assistant turns are cut to their head and tail.
    cutoff = assistant_turns[-_KEEP_FULL_TURNS] if len(assistant_turns) >= _KEEP_FULL_TURNS else 0
    soft = max(1000, profile.output_limit // 2)
    pruned = []
    for i, msg in enumerate(messages):
        text = msg.get("content") or ""
        if i < cutoff and msg["role"] == "tool" and len(text) > soft:
            msg = {**msg, "content": text[: soft // 2] + "\n...\n" + text[-soft // 2:]}
        pruned.append(msg)

    # Step 2: still over budget? The oldest results become one-line stubs,
    # never those from the latest round of tool calls.
    total = estimate_messages_tokens(pruned)
    latest_round = assistant_turns[-1] if assistant_turns else 0
    for i in range(latest_round):
        if total <= profile.context:
            break
        msg = pruned[i]
        if msg["role"] != "tool" or msg["content"].startswith("[cleared"):
            continue
        stub = {**msg, "content": f"[cleared to save context: {msg.get('name', 'tool')} output, "
                                  f"{len(msg['content'])} chars. Run the tool again if you need it.]"}
        total -= estimate_messages_tokens([msg]) - estimate_messages_tokens([stub])
        pruned[i] = stub
    return pruned


async def summarize_history(provider: Provider, conversation: list[dict], budget: int) -> None:
    """Replace the oldest turns of *conversation* with a short summary, in place.

    Keeps whole exchanges from the end until about half of *budget* is used,
    always keeps the latest exchange, and starts the kept part on a user turn.
    If the summary call fails, nothing is dropped.
    """
    # Walk back one exchange (two messages) at a time while the kept part fits.
    cut = len(conversation) - 2
    while cut >= 2 and estimate_messages_tokens(conversation[cut - 2:]) <= budget // 2:
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
            if not text.startswith(SUMMARY_MARKER):
                text = text[:_SUMMARY_INPUT_CHARS]
            lines.append(f"[{m['role']}]: {text}")
    request = [
        {"role": "system", "content": "Summarize in under 200 words. Keep file paths, error messages, code "
                                      "snippets and key decisions verbatim. Say what was done and what remains."},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        response = await provider.chat(request, tools=None, temperature=0.3)
    except Exception as e:
        log.warning("Failed to summarize history: %s", e)
        return
    # Small-model fix: a user message, not a second system message, because small
    # models anchor on one system prompt.
    summary = {"role": "user", "content": f"{SUMMARY_MARKER}\n{response.content or 'Earlier conversation.'}"}
    conversation[:] = [summary, *conversation[cut:]]
