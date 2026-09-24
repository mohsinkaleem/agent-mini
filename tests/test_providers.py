"""Tests for provider-specific quirks: local streaming order + Ollama vision."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_mini.providers.base import ModelInfo, parse_text_tool_calls
from agent_mini.providers.local import LocalProvider
from agent_mini.providers.ollama import OllamaProvider

# ── B2: Ollama vision — image_url parts must be flattened to `images` ────


def test_ollama_clean_messages_flattens_image_url_content():
    """OpenAI-style multi-part vision content is translated into Ollama's
    flat message format with an `images` field."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]
    out = OllamaProvider._clean_messages(msgs)
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "user"
    assert msg["content"] == "what's in this?"
    # Base64 payload stripped of `data:...;base64,` prefix.
    assert msg["images"] == ["AAAA"]


def test_ollama_clean_messages_passes_through_plain_text():
    """String content messages must not gain an `images` field."""
    out = OllamaProvider._clean_messages([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_ollama_clean_messages_native_tool_messages():
    """B3: assistant tool calls keep object arguments; results use role=tool."""
    out = OllamaProvider._clean_messages([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "ok"},
    ])
    assert out[0]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
    assert out[1] == {"role": "tool", "content": "ok", "tool_name": "read_file"}


def test_ollama_clean_messages_multiple_images():
    """Multiple images accumulate in the `images` list."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
            ],
        }
    ]
    out = OllamaProvider._clean_messages(msgs)
    assert out[0]["images"] == ["AAA", "BBB"]
    assert out[0]["content"] == "compare"


# ── B1: LocalProvider streaming — tool calls sorted by stream index ─────


def test_local_provider_sorts_tool_calls_by_index_not_id():
    """Streaming accumulates tool calls by integer `index`. Sorting by
    `id` (an opaque string) can scramble multi-call order when the
    server assigns non-monotonic ids."""
    # Simulate what LocalProvider.chat_stream builds internally: the model
    # emitted `zebra` first (index 0) and `alpha` second (index 1), but
    # server-assigned ids sort the other way.
    tool_calls_by_idx = {
        0: {"id": "call_zzz", "name": "zebra", "arguments": "{}"},
        1: {"id": "call_aaa", "name": "alpha", "arguments": "{}"},
    }
    # Correct sort: preserve emission order via the integer key.
    ordered = [v for _idx, v in sorted(tool_calls_by_idx.items())]
    assert [t["name"] for t in ordered] == ["zebra", "alpha"]

    # Buggy sort-by-id would reverse them, breaking the model's intent.
    bad = sorted(tool_calls_by_idx.values(), key=lambda x: x["id"])
    assert [t["name"] for t in bad] == ["alpha", "zebra"]


# ── B6: temperature handling ────────────────────────────────────────


@pytest.mark.parametrize("model", ["o1", "o3", "o4-mini", "gpt-5", "gpt-5-mini", "openai/gpt-5"])
def test_local_payload_omits_temperature_for_reasoning_models(model):
    payload = LocalProvider(model=model)._payload([], None, 0.7, stream=False)
    assert "temperature" not in payload


def test_local_payload_keeps_temperature_for_chat_models():
    payload = LocalProvider(model="gpt-4o")._payload([], None, 0.3, stream=False)
    assert payload["temperature"] == 0.3


def test_local_payload_omits_null_temperature():
    payload = LocalProvider(model="gpt-4o")._payload([], None, None, stream=False)
    assert "temperature" not in payload


def test_local_payload_reasoning_effort():
    provider = LocalProvider(model="o3", reasoning_effort="low")
    assert provider._payload([], None, None, stream=False)["reasoning_effort"] == "low"


def test_ollama_payload_omits_null_temperature():
    payload = OllamaProvider()._build_payload([], tools=None, temperature=None, stream=False)
    assert payload["options"] == {}


# ── B2: num_ctx ─────────────────────────────────────────────────────


def test_ollama_num_ctx_from_agent_or_config():
    provider = OllamaProvider()
    provider.context_window = 12288
    assert provider._build_payload([], None, None, stream=False)["options"] == {"num_ctx": 12288}
    pinned = OllamaProvider(num_ctx=4096, keep_alive="10m")
    pinned.context_window = 12288
    payload = pinned._build_payload([], None, None, stream=False)
    assert payload["options"] == {"num_ctx": 4096}
    assert payload["keep_alive"] == "10m"


# ── Wire-level tests with httpx.MockTransport ───────────────────────


def _mock(provider, handler):
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _ndjson(*chunks) -> str:
    return "\n".join(json.dumps(c) for c in chunks) + "\n"


async def test_ollama_stream_collects_split_tool_calls_and_usage():
    """B4: tool calls spread over chunks are all kept; the done chunk carries usage."""
    body = _ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "b"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "prompt_eval_count": 10, "eval_count": 5},
    )
    provider = _mock(OllamaProvider(), lambda r: httpx.Response(200, text=body))
    resp = await provider.chat_stream([], on_delta=AsyncMock())
    assert [tc.arguments["path"] for tc in resp.tool_calls] == ["a", "b"]
    assert len({tc.id for tc in resp.tool_calls}) == 2
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.parametrize("stream", [False, True])
async def test_ollama_error_body_and_hint(stream):
    """B7: the server's error text and a fix-it hint reach the user."""
    provider = _mock(
        OllamaProvider(model="nope:1b"),
        lambda r: httpx.Response(404, json={"error": "model 'nope:1b' not found"}),
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        if stream:
            await provider.chat_stream([], on_delta=AsyncMock())
        else:
            await provider.chat([])
    assert "model 'nope:1b' not found" in str(exc.value)
    assert "ollama pull nope:1b" in str(exc.value)


async def test_ollama_model_info():
    def handler(request):
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={
            "details": {"parameter_size": "8.0B"},
            "model_info": {"llama.context_length": 131072},
            "capabilities": ["completion", "tools"],
        })

    info = await _mock(OllamaProvider(), handler).model_info()
    assert info == ModelInfo(params_b=8.0, context_length=131072, capabilities=["completion", "tools"])


async def test_ollama_model_info_missing_model():
    provider = _mock(OllamaProvider(), lambda r: httpx.Response(404, json={"error": "not found"}))
    assert await provider.model_info() is None


async def test_local_stream_sse_variants():
    """B5: 'data:' without a space, a usage-only chunk, and reasoning_content."""
    lines = [
        'data:{"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data:{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}',
        'data:{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\": \\"x\\"}"}}]}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}',
        "data: [DONE]",
    ]
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text="\n\n".join(lines))

    thinking = AsyncMock()
    resp = await _mock(LocalProvider(), handler).chat_stream([], on_delta=AsyncMock(), on_thinking=thinking)
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert resp.content == "Hi"
    assert resp.thinking == "hmm"
    thinking.assert_awaited_once_with("hmm")
    assert resp.tool_calls[0].arguments == {"path": "x"}
    assert resp.tool_calls[0].id.startswith("call_")
    assert resp.usage["total_tokens"] == 10


# ── F4: tool calls written as text ──────────────────────────────────

_KNOWN = {"read_file", "shell_exec"}


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>',
        '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "a.py"}}]',
        'Let me look.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
        '{"name": "read_file", "parameters": {"path": "a.py"}}',
        '<|python_tag|>{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}',
    ],
)
def test_parse_text_tool_calls_formats(content):
    calls, _rest = parse_text_tool_calls(content, _KNOWN)
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})]


def test_parse_text_tool_calls_ignores_unknown_tools_and_examples():
    assert parse_text_tool_calls('{"name": "rm_everything", "arguments": {}}', _KNOWN)[0] == []
    essay = "Here is how tool calls look:\n" + "x" * 400 + (
        '\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    )
    assert parse_text_tool_calls(essay, _KNOWN)[0] == []
    assert parse_text_tool_calls("I used read_file earlier.", _KNOWN) == ([], "I used read_file earlier.")


def test_parse_text_tool_calls_returns_remaining_text():
    calls, rest = parse_text_tool_calls(
        'Checking.\n<tool_call>{"name": "shell_exec", "arguments": {"command": "ls"}}</tool_call>', _KNOWN
    )
    assert calls[0].name == "shell_exec"
    assert rest == "Checking."
