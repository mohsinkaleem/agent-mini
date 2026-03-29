"""Claude provider — Anthropic Messages API."""

from __future__ import annotations

import json

import httpx

from .base import BaseProvider, ChatResponse, StreamCallback, ToolCall, parse_arguments

_API_BASE = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"


class ClaudeProvider(BaseProvider):
    """Direct REST calls to the Anthropic Messages API.

    No SDK dependency — uses ``httpx`` with native tool-use support.
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return "claude"

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
        """Convert OpenAI-format messages → Anthropic messages + system."""
        system: str | None = None
        anthropic_msgs: list[dict] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                # Anthropic takes system as a top-level parameter
                system = msg["content"] if not system else system + "\n" + msg["content"]

            elif role == "user":
                content = msg["content"]
                if isinstance(content, list):
                    # Multi-modal content (vision)
                    parts: list[dict] = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append({"type": "text", "text": part["text"]})
                        elif part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                header, data = url.split(",", 1)
                                mime = header.split(":")[1].split(";")[0]
                                parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": data,
                                    },
                                })
                            else:
                                parts.append({
                                    "type": "image",
                                    "source": {"type": "url", "url": url},
                                })
                    anthropic_msgs.append({"role": "user", "content": parts})
                else:
                    anthropic_msgs.append({"role": "user", "content": content})

            elif role == "assistant":
                content_blocks: list[dict] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    func = tc["function"]
                    args = parse_arguments(func.get("arguments", {}))
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func["name"],
                        "input": args,
                    })
                if content_blocks:
                    anthropic_msgs.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                # Tool results in Anthropic format
                anthropic_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg["content"],
                    }],
                })

        return system, anthropic_msgs

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool defs → Anthropic tool format."""
        anthropic_tools = []
        for tool in tools:
            func = tool["function"]
            t: dict = {
                "name": func["name"],
                "description": func.get("description", ""),
            }
            if func.get("parameters"):
                t["input_schema"] = func["parameters"]
            else:
                t["input_schema"] = {"type": "object", "properties": {}}
            anthropic_tools.append(t)
        return anthropic_tools

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        system, anthropic_msgs = self._convert_messages(messages)

        payload: dict = {
            "model": self._model,
            "messages": anthropic_msgs,
            "max_tokens": 8192,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._convert_tools(tools)

        resp = await self._client.post(
            f"{_API_BASE}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        return self._parse_response(data)

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        system, anthropic_msgs = self._convert_messages(messages)

        payload: dict = {
            "model": self._model,
            "messages": anthropic_msgs,
            "max_tokens": 8192,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._convert_tools(tools)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Track the current tool_use block being streamed
        current_tool_id: str = ""
        current_tool_name: str = ""
        current_tool_input: list[str] = []
        usage: dict | None = None

        async with self._client.stream(
            "POST",
            f"{_API_BASE}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool_id = block.get("id", "")
                        current_tool_name = block.get("name", "")
                        current_tool_input = []

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            content_parts.append(text)
                            await on_delta(text)
                    elif delta.get("type") == "input_json_delta":
                        current_tool_input.append(delta.get("partial_json", ""))

                elif event_type == "content_block_stop":
                    if current_tool_name:
                        input_json = "".join(current_tool_input)
                        tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments=parse_arguments(input_json),
                        ))
                        current_tool_id = ""
                        current_tool_name = ""
                        current_tool_input = []

                elif event_type == "message_delta":
                    msg_usage = event.get("usage")
                    if msg_usage:
                        usage = {
                            "completion_tokens": msg_usage.get("output_tokens", 0),
                        }

                elif event_type == "message_start":
                    msg = event.get("message", {})
                    msg_usage = msg.get("usage")
                    if msg_usage:
                        usage = usage or {}
                        usage["prompt_tokens"] = msg_usage.get("input_tokens", 0)

        content = "".join(content_parts) or None

        # Finalize usage
        if usage:
            usage.setdefault("prompt_tokens", 0)
            usage.setdefault("completion_tokens", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return ChatResponse(
            content=content,
            tool_calls=tool_calls or None,
            usage=usage,
        )

    @staticmethod
    def _parse_response(data: dict) -> ChatResponse:
        """Parse an Anthropic Messages API response into ChatResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                content_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = {
                "prompt_tokens": raw_usage.get("input_tokens", 0),
                "completion_tokens": raw_usage.get("output_tokens", 0),
                "total_tokens": (
                    raw_usage.get("input_tokens", 0)
                    + raw_usage.get("output_tokens", 0)
                ),
            }

        return ChatResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls or None,
            usage=usage,
        )
