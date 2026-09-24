"""Ollama provider — local models via Ollama API."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .base import (
    INVALID_ARGS_KEY,
    BaseProvider,
    ChatResponse,
    ModelInfo,
    StreamCallback,
    parse_arguments,
    parse_openai_tool_calls,
    raise_for_status,
)

_PARAM_SIZE = re.compile(r"([\d.]+)\s*([KMBT]?)", re.IGNORECASE)
_SCALE = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3, "": 1e-9}


def _parse_param_size(text: str) -> float | None:
    """'8.0B' -> 8.0, '595.78M' -> 0.59578 (billions)."""
    m = _PARAM_SIZE.match(text.strip())
    if not m:
        return None
    return float(m.group(1)) * _SCALE[m.group(2).upper()]


def _usage(data: dict) -> dict | None:
    prompt, completion = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    if not (prompt or completion):
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class OllamaProvider(BaseProvider):
    """Connect to a running Ollama instance.

    Ollama exposes ``/api/chat`` with tool-calling, thinking, and streaming.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        think: bool | str | None = None,
        num_ctx: int | None = None,
        keep_alive: str | int | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._think = think
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._info: ModelInfo | None = None
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def num_ctx(self) -> int | None:
        """providers.ollama.numCtx if set, otherwise what the agent asked for."""
        return self._num_ctx or self.context_window

    def _hint(self, status: int, message: str) -> str:
        low = message.lower()
        if status == 404 and "not found" in low:
            return f"run: ollama pull {self._model}"
        if "does not support tools" in low:
            return "this model has no tool support; pick a tool-capable model (e.g. qwen3, llama3.1)"
        return ""

    async def model_info(self) -> ModelInfo | None:
        if self._info is not None:
            return self._info
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/show", json={"model": self._model}, timeout=10
            )
            if resp.status_code >= 400:
                return None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        details = data.get("details") or {}
        facts = data.get("model_info") or {}
        params = _parse_param_size(details.get("parameter_size") or "")
        if params is None and isinstance(facts.get("general.parameter_count"), int):
            params = facts["general.parameter_count"] / 1e9
        ctx = next(
            (v for k, v in facts.items() if k.endswith(".context_length") and isinstance(v, int)),
            None,
        )
        caps = data.get("capabilities")
        self._info = ModelInfo(
            params_b=params,
            context_length=ctx,
            capabilities=[str(c) for c in caps] if isinstance(caps, list) else None,
        )
        return self._info

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, stream=False
        )
        resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        await raise_for_status(resp, self._hint)
        return self._extract_chat_response(resp.json())

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float | None = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, stream=True
        )

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        last_message: dict[str, Any] = {}
        # Newer Ollama builds can spread tool calls over several chunks.
        tool_calls_raw: list[dict] = []
        usage = None

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
        ) as resp:
            await raise_for_status(resp, self._hint)
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
                    tool_calls_raw.extend(message["tool_calls"])
                if chunk.get("done"):
                    usage = _usage(chunk)

        content = "".join(content_parts) or (last_message.get("content") or None)
        thinking = "".join(thinking_parts) or (last_message.get("thinking") or None)
        tool_calls = parse_openai_tool_calls(tool_calls_raw)
        return ChatResponse(
            content=content, tool_calls=tool_calls, thinking=thinking, usage=usage
        )

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        stream: bool,
    ) -> dict:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        # Without num_ctx Ollama uses its small default and silently cuts the
        # front of the prompt, which is where the system prompt lives.
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        payload: dict = {
            "model": self._model,
            "messages": self._clean_messages(messages),
            "stream": stream,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if self._think not in (None, ""):
            payload["think"] = self._think
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload

    @staticmethod
    def _extract_chat_response(data: dict) -> ChatResponse:
        msg = data.get("message", {})
        return ChatResponse(
            content=msg.get("content") or None,
            tool_calls=parse_openai_tool_calls(msg.get("tool_calls")),
            thinking=msg.get("thinking") or None,
            usage=_usage(data),
        )

    @staticmethod
    def _clean_messages(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Ollama's format.

        Tool calls and results use Ollama's native shape (``tool_calls`` with
        object arguments, ``role: "tool"`` with ``tool_name``).

        Also translates OpenAI-style vision content parts
        (``[{"type": "image_url", "image_url": {"url": "..."}}, ...]``)
        into Ollama's flat ``images: ["<base64>"]`` field on the message,
        so vision works on the default provider.
        """
        cleaned = []
        for msg in messages:
            if msg["role"] == "tool":
                cleaned.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_name": msg.get("name", ""),
                })
                continue

            base = {k: v for k, v in msg.items() if k != "tool_calls"}
            if msg.get("tool_calls"):
                calls = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = parse_arguments(func.get("arguments", {}))
                    if INVALID_ARGS_KEY in args:
                        args = {}
                    calls.append({"function": {"name": func.get("name", ""), "arguments": args}})
                base["tool_calls"] = calls
            content = base.get("content")

            # OpenAI-style multi-part content → Ollama flat {content, images}.
            if isinstance(content, list):
                texts: list[str] = []
                images: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        # Ollama's `images` accepts either a raw base64 string
                        # or a URL. Strip the `data:...;base64,` prefix if
                        # present so it works with older builds too.
                        if url.startswith("data:") and ";base64," in url:
                            url = url.split(";base64,", 1)[1]
                        if url:
                            images.append(url)
                base["content"] = "\n".join(t for t in texts if t)
                if images:
                    base["images"] = images
            cleaned.append(base)
        return cleaned
