"""Local / custom OpenAI-compatible provider.

Works with any server that implements the OpenAI chat completions API:
LM Studio, vLLM, llama.cpp, text-generation-webui, Oobabooga, etc.
"""

from __future__ import annotations

import httpx

from .base import BaseProvider, ChatResponse, parse_openai_tool_calls


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

        return ChatResponse(content=content, tool_calls=tool_calls)
