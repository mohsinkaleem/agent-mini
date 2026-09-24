"""Local / custom OpenAI-compatible provider.

Works with any server that implements the OpenAI chat completions API:
LM Studio, vLLM, llama.cpp, text-generation-webui, Oobabooga, etc.
"""

from __future__ import annotations

import json
import re

import httpx

from .base import (
    BaseProvider,
    ChatResponse,
    StreamCallback,
    ToolCall,
    new_call_id,
    parse_arguments,
    parse_openai_tool_calls,
    raise_for_status,
)

# OpenAI reasoning models return 400 for any temperature other than the default.
_NO_TEMPERATURE_MODELS = re.compile(r"^(?:o\d|gpt-5)", re.IGNORECASE)


class LocalProvider(BaseProvider):
    """Generic OpenAI-compatible endpoint for self-hosted models."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "no-key",
        model: str = "local-model",
        reasoning_effort: str | None = None,
        name: str = "local",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._name = name
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def _payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        stream: bool,
    ) -> dict:
        payload: dict = {"model": self._model, "messages": messages, "stream": stream}
        model = self._model.rsplit("/", 1)[-1]
        if temperature is not None and not _NO_TEMPERATURE_MODELS.match(model):
            payload["temperature"] = temperature
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
    ) -> ChatResponse:
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=self._payload(messages, tools, temperature, stream=False),
        )
        await raise_for_status(resp)
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or None
        tool_calls = parse_openai_tool_calls(msg.get("tool_calls"))
        usage = data.get("usage")

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            # vLLM, llama.cpp and DeepSeek put thinking here.
            thinking=msg.get("reasoning_content") or None,
            usage=usage,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict] = {}
        usage = None

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=self._payload(messages, tools, temperature, stream=True),
        ) as resp:
            await raise_for_status(resp)
            async for line in resp.aiter_lines():
                # Some servers send "data:{...}" without the space.
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}

                # Text content
                text = delta.get("content") or ""
                if text:
                    content_parts.append(text)
                    await on_delta(text)

                thinking = delta.get("reasoning_content") or ""
                if thinking:
                    thinking_parts.append(thinking)
                    if on_thinking:
                        await on_thinking(thinking)

                # Accumulate tool call chunks
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": tc.get("id") or new_call_id(),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                    if tc.get("function", {}).get("name"):
                        tool_calls_by_idx[idx]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_calls_by_idx[idx]["arguments"] += tc["function"]["arguments"]
                    if tc.get("id"):
                        tool_calls_by_idx[idx]["id"] = tc["id"]

        content = "".join(content_parts) or None
        tcs: list[ToolCall] | None = None
        if tool_calls_by_idx:
            # Sort by the integer stream index — the true order the model
            # emitted the calls in. Sorting by `id` (an opaque string) can
            # scramble order (e.g. "call_10" < "call_2" lexicographically).
            tcs = [
                ToolCall(
                    id=v["id"],
                    name=v["name"],
                    arguments=parse_arguments(v["arguments"]),
                )
                for _idx, v in sorted(tool_calls_by_idx.items())
            ]
        return ChatResponse(
            content=content,
            tool_calls=tcs,
            thinking="".join(thinking_parts) or None,
            usage=usage,
        )
