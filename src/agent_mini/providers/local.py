"""Local / custom OpenAI-compatible provider.

Works with any server that implements the OpenAI chat completions API:
LM Studio, vLLM, llama.cpp, text-generation-webui, Oobabooga, etc.
"""

from __future__ import annotations

import json

import httpx

from .base import (
    BaseProvider,
    ChatResponse,
    StreamCallback,
    ToolCall,
    parse_arguments,
    parse_openai_tool_calls,
)


class LocalProvider(BaseProvider):
    """Generic OpenAI-compatible endpoint for self-hosted models."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "no-key",
        model: str = "local-model",
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or None
        tool_calls = parse_openai_tool_calls(msg.get("tool_calls"))
        usage = data.get("usage")

        return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        content_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict] = {}

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Text content
                text = delta.get("content") or ""
                if text:
                    content_parts.append(text)
                    await on_delta(text)

                # Accumulate tool call chunks
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": tc.get("id", f"call_{idx}"),
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
            tcs = [
                ToolCall(
                    id=v["id"],
                    name=v["name"],
                    arguments=parse_arguments(v["arguments"]),
                )
                for v in sorted(tool_calls_by_idx.values(), key=lambda x: x["id"])
            ]
        return ChatResponse(content=content, tool_calls=tcs)
