"""Ollama provider — local models via Ollama API."""

from __future__ import annotations

import json
import httpx

from .base import BaseProvider, ChatResponse, ToolCall


class OllamaProvider(BaseProvider):
    """Connect to a running Ollama instance.

    Ollama exposes an OpenAI-compatible ``/api/chat`` endpoint with native
    tool-calling support for compatible models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
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
        payload: dict = {
            "model": self._model,
            "messages": self._clean_messages(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        content = msg.get("content") or None
        tool_calls = None

        if msg.get("tool_calls"):
            tool_calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

        return ChatResponse(content=content, tool_calls=tool_calls)

    @staticmethod
    def _clean_messages(messages: list[dict]) -> list[dict]:
        """Ollama expects a flat message list; convert tool results."""
        cleaned = []
        for msg in messages:
            if msg["role"] == "tool":
                # Ollama doesn't support tool role natively in older versions —
                # wrap as a user message with context.
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
                # Strip tool_calls key from assistant messages for compat
                cleaned.append(
                    {k: v for k, v in msg.items() if k != "tool_calls"}
                )
        return cleaned
