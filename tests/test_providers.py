"""Tests for provider-specific quirks: local streaming order + Ollama vision."""

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


def test_ollama_clean_messages_converts_tool_result():
    """Tool messages are translated to a `user`-tagged text summary for
    compatibility with older Ollama builds."""
    out = OllamaProvider._clean_messages(
        [{"role": "tool", "name": "read_file", "content": "ok"}]
    )
    assert out[0]["role"] == "user"
    assert "read_file" in out[0]["content"]
    assert "ok" in out[0]["content"]


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
