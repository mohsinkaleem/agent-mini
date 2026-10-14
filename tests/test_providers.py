"""Providers: wire formats, streaming, errors, and tool calls written as text."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_mini.config import with_defaults
from agent_mini.providers import OllamaProvider, OpenAIProvider, create_provider
from agent_mini.providers.base import ModelInfo, parse_text_tool_calls
from agent_mini.providers.ollama import to_ollama


def _mock(provider, handler):
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _ndjson(*chunks) -> str:
    return "\n".join(json.dumps(c) for c in chunks) + "\n"


# ── Ollama ───────────────────────────────────────────────────────────


def test_to_ollama_messages():
    out = to_ollama([
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "https://x.org/b.png"}},
        ]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "ok"},
    ])
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "user", "content": "compare", "images": ["AAA", "https://x.org/b.png"]}
    assert out[2]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
    assert out[3] == {"role": "tool", "content": "ok", "tool_name": "read_file"}


def test_ollama_payload():
    provider = OllamaProvider("qwen3:8b")
    assert provider.payload([], None, None)["options"] == {"num_predict": 4096}  # no endless generation
    provider.context_window = 12288
    payload = provider.payload([], [{"t": 1}], 0.2)
    assert payload["options"] == {"num_predict": 4096, "temperature": 0.2, "num_ctx": 12288}
    assert payload["think"] is False and payload["stream"] is True and payload["tools"] == [{"t": 1}]
    pinned = OllamaProvider("qwen3:8b", num_ctx=4096, keep_alive="10m", think=None)
    pinned.context_window = 12288
    payload = pinned.payload([], None, None)
    assert payload["options"]["num_ctx"] == 4096 and payload["keep_alive"] == "10m" and "think" not in payload


async def test_ollama_streams_text_tool_calls_and_usage():
    body = _ndjson(
        {"message": {"role": "assistant", "thinking": "hmm"}, "done": False},
        {"message": {"role": "assistant", "content": "Look"}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "b"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 5},
    )
    provider = _mock(OllamaProvider("qwen3:8b"), lambda r: httpx.Response(200, text=body))
    on_delta, on_thinking = AsyncMock(), AsyncMock()
    resp = await provider.chat([], on_delta=on_delta, on_thinking=on_thinking)
    assert resp.content == "Look" and resp.thinking == "hmm"
    on_delta.assert_awaited_once_with("Look")
    on_thinking.assert_awaited_once_with("hmm")
    assert [tc.arguments["path"] for tc in resp.tool_calls] == ["a", "b"]
    assert len({tc.id for tc in resp.tool_calls}) == 2
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert (await provider.chat([])).content == "Look"  # no callbacks needed


async def test_ollama_error_body_and_hint():
    error = {"error": "model 'nope:1b' not found"}
    provider = _mock(OllamaProvider("nope:1b"), lambda r: httpx.Response(404, json=error))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await provider.chat([])
    assert "model 'nope:1b' not found" in str(exc.value) and "ollama pull nope:1b" in str(exc.value)


async def test_ollama_model_info():
    def handler(request):
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"details": {"parameter_size": "8.0B"},
                                         "model_info": {"llama.context_length": 131072},
                                         "capabilities": ["completion", "tools"]})

    assert await _mock(OllamaProvider("m"), handler).model_info() == ModelInfo(8.0, 131072, ["completion", "tools"])
    missing = _mock(OllamaProvider("m"), lambda r: httpx.Response(404, json={"error": "not found"}))
    assert await missing.model_info() is None


# ── OpenAI-compatible ────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["o1", "o3", "o4-mini", "gpt-5", "gpt-5-mini", "openai/gpt-5"])
def test_reasoning_models_get_no_temperature(model):
    assert "temperature" not in OpenAIProvider(model, "http://x").payload([], None, 0.7, stream=False)


def test_openai_payload():
    provider = OpenAIProvider("gpt-4o", "http://x", reasoning_effort="low", name="openai")
    payload = provider.payload([], None, 0.3, stream=True)
    assert payload["temperature"] == 0.3 and payload["reasoning_effort"] == "low"
    assert payload["stream_options"] == {"include_usage": True}
    assert "temperature" not in provider.payload([], None, None, stream=False)
    assert "max_tokens" not in payload  # OpenAI's reasoning models count their thinking in it
    assert OpenAIProvider("m", "http://x").payload([], None, None, stream=False)["max_tokens"] == 4096


def test_tool_call_without_a_name():
    from agent_mini.providers.base import parse_openai_tool_calls
    assert parse_openai_tool_calls([{"function": {"name": None, "arguments": "{}"}}])[0].name == ""


async def test_openai_without_callback_does_not_stream():
    def handler(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi", "tool_calls": [
            {"id": "c", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}}],
            "usage": {"total_tokens": 3}})

    resp = await _mock(OpenAIProvider("m", "http://x"), handler).chat([])
    assert resp.content == "hi" and resp.tool_calls[0].arguments == {"path": "a"} and resp.usage["total_tokens"] == 3


async def test_openai_stream_variants():
    """'data:' without a space, split tool-call arguments, index order, a usage-only chunk, reasoning_content."""
    lines = [
        'data:{"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data:{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_aaa","function":{"name":"b","arguments":"{}"}}]}}]}',
        'data:{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}',
        'data:{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\": \\"x\\"}"}}]}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}',
        "data: [DONE]",
    ]
    on_thinking = AsyncMock()
    provider = _mock(OpenAIProvider("m", "http://x"), lambda r: httpx.Response(200, text="\n\n".join(lines)))
    resp = await provider.chat([], on_delta=AsyncMock(), on_thinking=on_thinking)
    assert resp.content == "Hi" and resp.thinking == "hmm"
    on_thinking.assert_awaited_once_with("hmm")
    assert [tc.name for tc in resp.tool_calls] == ["read_file", "b"]  # stream index order, not id order
    assert resp.tool_calls[0].arguments == {"path": "x"} and resp.tool_calls[0].id.startswith("call_")
    assert resp.usage["total_tokens"] == 10


def test_create_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    openai = create_provider(with_defaults({"provider": "openai"}))
    assert openai.name == "openai" and openai.base_url == "https://api.openai.com/v1"
    assert openai.headers["Authorization"] == "Bearer sk-env"
    local = create_provider(with_defaults({"provider": "local", "providers": {"local": {"model": "m"}}}))
    assert local.headers["Authorization"] == "Bearer no-key"  # OPENAI_API_KEY is not sent to other servers
    monkeypatch.setenv("AGENT_MINI_API_KEY", "am-env")
    assert create_provider(with_defaults({"provider": "local"})).headers["Authorization"] == "Bearer am-env"
    ollama = create_provider(with_defaults({}))
    assert ollama.name == "ollama" and ollama.model == "qwen3.5:9b"


# ── Tool calls written as text ───────────────────────────────────────

_KNOWN = {"read_file", "shell"}


@pytest.mark.parametrize("content", [
    '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>',
    '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "a.py"}}]',
    'Let me look.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
    '{"name": "read_file", "parameters": {"path": "a.py"}}',
    '<|python_tag|>{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}',
])
def test_parse_text_tool_calls_formats(content):
    calls, _ = parse_text_tool_calls(content, _KNOWN)
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})]


def test_parse_text_tool_calls_ignores_unknown_tools_and_examples():
    assert parse_text_tool_calls('{"name": "rm_everything", "arguments": {}}', _KNOWN)[0] == []
    essay = "Here is how tool calls look:\n" + "x" * 400 + \
        '\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    assert parse_text_tool_calls(essay, _KNOWN)[0] == []
    assert parse_text_tool_calls("I used read_file earlier.", _KNOWN) == ([], "I used read_file earlier.")


def test_parse_text_tool_calls_returns_remaining_text():
    calls, rest = parse_text_tool_calls(
        'Checking.\n<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>', _KNOWN)
    assert calls[0].name == "shell" and rest == "Checking."
