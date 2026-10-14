"""Ollama, through its native /api/chat (needed to set num_ctx)."""

from __future__ import annotations

import json
import re

import httpx

from ..prompt import MAX_REPLY_TOKENS
from .base import (
    INVALID_ARGS_KEY,
    ChatResponse,
    ModelInfo,
    Provider,
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
    return float(m.group(1)) * _SCALE[m.group(2).upper()] if m else None


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", think: bool | str | None = False,
                 num_ctx: int | None = None, keep_alive: str | int | None = None):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self._think = think
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._info: ModelInfo | None = None
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def num_ctx(self) -> int | None:
        """providers.ollama.numCtx if set, otherwise what the agent asked for."""
        return self._num_ctx or self.context_window

    def _hint(self, status: int, message: str) -> str:
        low = message.lower()
        if status == 404 and "not found" in low:
            return f"run: ollama pull {self.model}"
        if "does not support tools" in low:
            return "this model has no tool support; pick a tool-capable model (e.g. qwen3.5, llama3.1)"
        return ""

    async def model_info(self) -> ModelInfo | None:
        if self._info is not None:
            return self._info
        try:
            resp = await self._client.post(f"{self.base_url}/api/show", json={"model": self.model}, timeout=10)
            if resp.status_code >= 400:
                return None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        facts = data.get("model_info") or {}
        params = _parse_param_size((data.get("details") or {}).get("parameter_size") or "")
        if params is None and isinstance(facts.get("general.parameter_count"), int):
            params = facts["general.parameter_count"] / 1e9
        ctx = next((v for k, v in facts.items() if k.endswith(".context_length") and isinstance(v, int)), None)
        caps = data.get("capabilities")
        self._info = ModelInfo(params, ctx, [str(c) for c in caps] if isinstance(caps, list) else None)
        return self._info

    def payload(self, messages: list[dict], tools: list[dict] | None, temperature: float | None) -> dict:
        options: dict = {"num_predict": MAX_REPLY_TOKENS}
        if temperature is not None:
            options["temperature"] = temperature
        # Without num_ctx Ollama uses its small default and silently cuts the
        # front of the prompt, which is where the system prompt lives.
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        payload: dict = {"model": self.model, "messages": to_ollama(messages), "stream": True, "options": options}
        if tools:
            payload["tools"] = tools
        if self._think not in (None, ""):
            payload["think"] = self._think
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload

    async def chat(self, messages, tools=None, temperature=None, on_delta=None, on_thinking=None) -> ChatResponse:
        content, thinking, raw_calls = [], [], []
        usage = None
        async with self._client.stream("POST", f"{self.base_url}/api/chat",
                                       json=self.payload(messages, tools, temperature)) as resp:
            await raise_for_status(resp, self._hint)
            async for line in resp.aiter_lines():
                try:
                    chunk = json.loads(line) if line else {}
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message") or {}
                await _take(message.get("content"), content, on_delta)
                await _take(message.get("thinking"), thinking, on_thinking)
                # Newer Ollama builds can spread tool calls over several chunks.
                raw_calls.extend(message.get("tool_calls") or [])
                if chunk.get("done"):
                    prompt, completion = chunk.get("prompt_eval_count", 0), chunk.get("eval_count", 0)
                    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                             "total_tokens": prompt + completion}
        return ChatResponse(
            content="".join(content) or None,
            tool_calls=parse_openai_tool_calls(raw_calls),
            thinking="".join(thinking) or None,
            usage=usage,
        )


async def _take(delta: str | None, parts: list[str], callback: StreamCallback | None) -> None:
    if delta:
        parts.append(delta)
        if callback:
            await callback(delta)


def to_ollama(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages in Ollama's shape.

    Tool calls carry object arguments, tool results name their tool, and
    image_url content parts become the flat ``images: [<base64 or URL>]`` field.
    """
    out = []
    for msg in messages:
        if msg["role"] == "tool":
            out.append({"role": "tool", "content": msg.get("content", ""), "tool_name": msg.get("name", "")})
            continue
        base = {k: v for k, v in msg.items() if k != "tool_calls"}
        if msg.get("tool_calls"):
            base["tool_calls"] = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = parse_arguments(func.get("arguments", {}))
                base["tool_calls"].append(
                    {"function": {"name": func.get("name", ""), "arguments": {} if INVALID_ARGS_KEY in args else args}}
                )
        if isinstance(base.get("content"), list):
            parts = base["content"]
            base["content"] = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text"))
            images = [(p.get("image_url") or {}).get("url", "") for p in parts if p.get("type") == "image_url"]
            # Strip the data: prefix so older builds that want raw base64 work too.
            images = [u.split(";base64,", 1)[1] if u.startswith("data:") and ";base64," in u else u for u in images]
            if images := [u for u in images if u]:
                base["images"] = images
        out.append(base)
    return out
