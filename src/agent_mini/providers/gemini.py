"""Gemini provider — Google Generative AI REST API."""

from __future__ import annotations

import json

import httpx

from .base import BaseProvider, ChatResponse, StreamCallback, ToolCall, parse_arguments

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseProvider):
    """Direct REST calls to the Gemini API.

    No SDK dependency — uses ``httpx`` with Gemini's ``generateContent``
    endpoint including native function-calling support.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages → Gemini ``contents`` + system."""
        system: str | None = None
        contents: list[dict] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system = msg["content"]

            elif role == "user":
                content = msg["content"]
                if isinstance(content, list):
                    # Multi-modal content (vision)
                    gemini_parts: list[dict] = []
                    for part in content:
                        if part.get("type") == "text":
                            gemini_parts.append({"text": part["text"]})
                        elif part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                # Inline base64 image
                                # Format: data:mime;base64,DATA
                                header, data = url.split(",", 1)
                                mime = header.split(":")[1].split(";")[0]
                                gemini_parts.append({
                                    "inlineData": {
                                        "mimeType": mime,
                                        "data": data,
                                    }
                                })
                            else:
                                # URL-based image — Gemini needs inline data,
                                # but we pass as text fallback
                                gemini_parts.append({"text": f"[Image: {url}]"})
                    contents.append({"role": "user", "parts": gemini_parts})
                else:
                    contents.append({"role": "user", "parts": [{"text": content}]})

            elif role == "assistant":
                parts: list[dict] = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    func = tc["function"]
                    args = parse_arguments(func.get("arguments", {}))
                    parts.append({"functionCall": {"name": func["name"], "args": args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})

            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("name", "tool"),
                                    "response": {"result": msg["content"]},
                                }
                            }
                        ],
                    }
                )

        return system, contents

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool defs → Gemini ``functionDeclarations``."""
        declarations = []
        for tool in tools:
            func = tool["function"]
            decl: dict = {
                "name": func["name"],
                "description": func.get("description", ""),
            }
            if func.get("parameters"):
                decl["parameters"] = func["parameters"]
            declarations.append(decl)
        return [{"function_declarations": declarations}]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        system, contents = self._convert_messages(messages)

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._convert_tools(tools)

        url = f"{_API_BASE}/models/{self._model}:generateContent?key={self._api_key}"
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        content: str | None = None
        tool_calls: list[ToolCall] | None = None

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts: list[str] = []
            tcs: list[ToolCall] = []

            for i, part in enumerate(parts):
                if "text" in part:
                    texts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tcs.append(
                        ToolCall(
                            id=f"call_{i}",
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )

            if texts:
                content = "\n".join(texts)
            if tcs:
                tool_calls = tcs

        # Extract usage metadata from Gemini response
        usage = None
        usage_meta = data.get("usageMetadata")
        if usage_meta:
            usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            }

        return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        system, contents = self._convert_messages(messages)

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._convert_tools(tools)

        url = (
            f"{_API_BASE}/models/{self._model}:streamGenerateContent"
            f"?alt=sse&key={self._api_key}"
        )

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tc_index = 0

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if not raw.strip():
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                candidates = chunk.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        delta = part["text"]
                        content_parts.append(delta)
                        await on_delta(delta)
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{tc_index}",
                                name=fc["name"],
                                arguments=fc.get("args", {}),
                            )
                        )
                        tc_index += 1

        content = "".join(content_parts) or None
        return ChatResponse(
            content=content,
            tool_calls=tool_calls or None,
        )
