"""Ollama provider — local models via Ollama API."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import (
    BaseProvider,
    ChatResponse,
    StreamCallback,
    parse_openai_tool_calls,
)


class OllamaProvider(BaseProvider):
    """Connect to a running Ollama instance.

    Ollama exposes ``/api/chat`` with tool-calling, thinking, and streaming.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        think: bool | str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._think = think
        self._client = httpx.AsyncClient(timeout=300)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, stream=False
        )
        resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return self._extract_chat_response(resp.json())

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, stream=True
        )

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        last_message: dict[str, Any] = {}
        tool_calls_raw: list[dict] | None = None

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = chunk.get("message") or {}
                if message:
                    last_message = message

                delta = message.get("content") or ""
                if delta:
                    content_parts.append(delta)
                    await on_delta(delta)

                thinking_delta = message.get("thinking") or ""
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    if on_thinking:
                        await on_thinking(thinking_delta)

                if message.get("tool_calls"):
                    tool_calls_raw = message["tool_calls"]

        content = "".join(content_parts) or (last_message.get("content") or None)
        thinking = "".join(thinking_parts) or (last_message.get("thinking") or None)
        tool_calls = parse_openai_tool_calls(
            tool_calls_raw or last_message.get("tool_calls")
        )
        return ChatResponse(content=content, tool_calls=tool_calls, thinking=thinking)

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": self._clean_messages(messages),
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        if self._think not in (None, ""):
            payload["think"] = self._think
        return payload

    @staticmethod
    def _extract_chat_response(data: dict) -> ChatResponse:
        msg = data.get("message", {})
        return ChatResponse(
            content=msg.get("content") or None,
            tool_calls=parse_openai_tool_calls(msg.get("tool_calls")),
            thinking=msg.get("thinking") or None,
        )

    @staticmethod
    def _clean_messages(messages: list[dict]) -> list[dict]:
        """Ollama expects a flat message list; convert tool results."""
        cleaned = []
        for msg in messages:
            if msg["role"] == "tool":
                # Convert tool results for compatibility with older Ollama builds.
                cleaned.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Tool result for {msg.get('name', 'tool')}]:\n"
                            f"{msg['content']}"
                        ),
                    }
                )
            else:
                cleaned.append({k: v for k, v in msg.items() if k != "tool_calls"})
        return cleaned
