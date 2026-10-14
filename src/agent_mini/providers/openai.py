"""OpenAI's API and every server that speaks it: LM Studio, vLLM, llama.cpp, …"""

from __future__ import annotations

import json
import re

import httpx

from ..prompt import MAX_REPLY_TOKENS
from .base import (
    ChatResponse,
    Provider,
    ToolCall,
    new_call_id,
    parse_arguments,
    parse_openai_tool_calls,
    raise_for_status,
)

# OpenAI reasoning models return 400 for any temperature other than the default.
_NO_TEMPERATURE = re.compile(r"^(?:o\d|gpt-5)", re.IGNORECASE)


class OpenAIProvider(Provider):
    def __init__(self, model: str, base_url: str, api_key: str = "", reasoning_effort: str | None = None,
                 name: str = "local"):
        super().__init__(model)
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key or 'no-key'}", "Content-Type": "application/json"}
        self._reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    def payload(self, messages: list[dict], tools: list[dict] | None, temperature: float | None, stream: bool) -> dict:
        payload: dict = {"model": self.model, "messages": messages, "stream": stream}
        if temperature is not None and not _NO_TEMPERATURE.match(self.model.rsplit("/", 1)[-1]):
            payload["temperature"] = temperature
        if self.name == "local":  # OpenAI's reasoning models count thinking in this; local servers don't cap at all
            payload["max_tokens"] = MAX_REPLY_TOKENS
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, messages, tools=None, temperature=None, on_delta=None, on_thinking=None) -> ChatResponse:
        # Stream only when someone is watching: some servers can't stream tool calls.
        if on_delta is not None:
            return await self._stream(messages, tools, temperature, on_delta, on_thinking)
        resp = await self._client.post(f"{self.base_url}/chat/completions", headers=self.headers,
                                       json=self.payload(messages, tools, temperature, stream=False))
        await raise_for_status(resp)
        data = resp.json()
        msg = data["choices"][0]["message"]
        return ChatResponse(
            content=msg.get("content") or None,
            tool_calls=parse_openai_tool_calls(msg.get("tool_calls")),
            # vLLM, llama.cpp and DeepSeek put thinking here.
            thinking=msg.get("reasoning_content") or None,
            usage=data.get("usage"),
        )

    async def _stream(self, messages, tools, temperature, on_delta, on_thinking) -> ChatResponse:
        content, thinking = [], []
        calls: dict[int, dict] = {}
        usage = None
        async with self._client.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers,
                                       json=self.payload(messages, tools, temperature, stream=True)) as resp:
            await raise_for_status(resp)
            async for line in resp.aiter_lines():
                # Some servers send "data:{...}" without the space.
                raw = line[len("data:"):].strip() if line.startswith("data:") else ""
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or usage
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if text := delta.get("content"):
                    content.append(text)
                    await on_delta(text)
                if thought := delta.get("reasoning_content"):
                    thinking.append(thought)
                    if on_thinking:
                        await on_thinking(thought)
                for tc in delta.get("tool_calls") or []:
                    call = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    func = tc.get("function") or {}
                    call["id"] = tc.get("id") or call["id"]
                    call["name"] = func.get("name") or call["name"]
                    call["arguments"] += func.get("arguments") or ""
        # Order by stream index, the order the model emitted them ("call_10" < "call_2" as strings).
        tool_calls = [ToolCall(c["id"] or new_call_id(), c["name"], parse_arguments(c["arguments"]))
                      for _, c in sorted(calls.items())]
        return ChatResponse("".join(content) or None, tool_calls or None, "".join(thinking) or None, usage)
