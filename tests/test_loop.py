"""Tests for AgentLoop — parallel execution, self-reflection, retry."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_mini.agent.loop import AgentLoop
from agent_mini.agent.memory import Memory
from agent_mini.agent.token_estimator import estimate_messages_tokens, num_ctx_for
from agent_mini.providers.base import ChatResponse, ModelInfo, ToolCall

FakeResponse, FakeToolCall = ChatResponse, ToolCall


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "mem.json")


@pytest.fixture
def config(tmp_path: Path) -> dict:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return {
        "workspace": str(ws),
        "tools": {"restrictToWorkspace": False},
        "agent": {"maxIterations": 5},
    }


def _make_agent(config, memory, responses):
    """Create an AgentLoop with a mock provider that returns a sequence of responses."""
    provider = AsyncMock()
    provider.chat = AsyncMock(side_effect=responses)
    provider.chat_stream = AsyncMock(side_effect=responses)
    provider.close = AsyncMock()
    provider.model_name = "qwen2.5:7b"  # default to small tier for tests
    return AgentLoop(provider, config, memory)


@pytest.mark.asyncio
async def test_simple_text_response(config, memory):
    agent = _make_agent(config, memory, [FakeResponse(content="Hello!")])
    conversation = []
    result = await agent.run("hi", conversation)
    assert result == "Hello!"
    assert len(conversation) == 2  # user + assistant
    await agent.close()


@pytest.mark.asyncio
async def test_tool_call_then_text(config, memory):
    """Agent calls a tool, gets result, then responds with text."""
    ws = Path(config["workspace"])
    (ws / "test.txt").write_text("file contents")

    responses = [
        FakeResponse(
            content="",
            tool_calls=[
                FakeToolCall(id="tc1", name="read_file", arguments={"path": "test.txt"})
            ],
        ),
        FakeResponse(content="The file says: file contents"),
    ]
    agent = _make_agent(config, memory, responses)
    conversation = []
    result = await agent.run("read test.txt", conversation)
    assert "file contents" in result
    await agent.close()


@pytest.mark.asyncio
async def test_parallel_tool_execution(config, memory):
    """Multiple tool calls should be executed concurrently."""
    ws = Path(config["workspace"])
    (ws / "a.txt").write_text("aaa")
    (ws / "b.txt").write_text("bbb")

    responses = [
        FakeResponse(
            content="",
            tool_calls=[
                FakeToolCall(id="tc1", name="read_file", arguments={"path": "a.txt"}),
                FakeToolCall(id="tc2", name="read_file", arguments={"path": "b.txt"}),
            ],
        ),
        FakeResponse(content="Got both files"),
    ]
    agent = _make_agent(config, memory, responses)
    conversation = []
    result = await agent.run("read both files", conversation)
    assert result == "Got both files"

    # Verify both tool results were sent to the provider
    second_call_messages = agent.provider.chat.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "aaa" in tool_msgs[0]["content"]
    assert "bbb" in tool_msgs[1]["content"]
    await agent.close()


@pytest.mark.asyncio
async def test_self_reflection_on_error(config, memory):
    """Tool errors should be wrapped with a reflection prompt."""
    responses = [
        FakeResponse(
            content="",
            tool_calls=[
                FakeToolCall(
                    id="tc1",
                    name="read_file",
                    arguments={"path": "nonexistent.txt"},
                )
            ],
        ),
        FakeResponse(content="File not found, let me try differently"),
    ]
    agent = _make_agent(config, memory, responses)
    conversation = []
    await agent.run("read it", conversation)

    # Check reflection was added to tool result
    second_call_messages = agent.provider.chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m.get("role") == "tool"][0]
    assert "different approach" in tool_msg["content"]
    await agent.close()


@pytest.mark.asyncio
async def test_max_iterations(config, memory):
    """Agent should stop after max iterations."""
    config["agent"]["maxIterations"] = 2

    # Always return tool calls, never text — should hit max iterations
    tool_response = FakeResponse(
        content="",
        tool_calls=[
            FakeToolCall(id="tc1", name="list_directory", arguments={"path": "."})
        ],
    )
    responses = [tool_response] * 5
    agent = _make_agent(config, memory, responses)
    conversation = []
    result = await agent.run("loop forever", conversation)
    assert "max iterations" in result.lower()
    await agent.close()


@pytest.mark.asyncio
async def test_max_iterations_persists_turn(config, memory):
    """B4: reaching max iterations must still leave a record of the turn
    in *conversation* so session history matches what the user sees."""
    config["agent"]["maxIterations"] = 2
    tool_response = FakeResponse(
        content="",
        tool_calls=[
            FakeToolCall(id="tc1", name="list_directory", arguments={"path": "."})
        ],
    )
    responses = [tool_response] * 5
    agent = _make_agent(config, memory, responses)
    conversation: list[dict] = []
    await agent.run("keep looping", conversation)

    # Both sides of the turn must be persisted.
    assert len(conversation) == 2
    assert conversation[0] == {"role": "user", "content": "keep looping"}
    assert conversation[1]["role"] == "assistant"
    assert "max iterations" in conversation[1]["content"].lower()
    await agent.close()


@pytest.mark.asyncio
async def test_retry_on_transient_error(config, memory):
    """Transient HTTP errors should be retried."""
    # First call fails with 503, second succeeds
    mock_response = MagicMock()
    mock_response.status_code = 503

    provider = AsyncMock()
    provider.chat = AsyncMock(
        side_effect=[
            httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_response),
            FakeResponse(content="Recovered!"),
        ]
    )
    provider.close = AsyncMock()
    provider.model_name = "qwen2.5:7b"

    agent = AgentLoop(provider, config, memory)
    conversation = []

    # Patch sleep to avoid waiting
    with patch("agent_mini.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await agent.run("test retry", conversation)

    assert result == "Recovered!"
    assert provider.chat.call_count == 2
    await agent.close()


@pytest.mark.asyncio
async def test_summarize_history_failure_preserves_data(config, memory):
    """If summarization fails, conversation should remain intact."""
    provider = AsyncMock()
    # Summarization call always fails
    provider.chat = AsyncMock(side_effect=Exception("Summarization failed!"))
    provider.close = AsyncMock()
    provider.model_name = "qwen2.5:7b"

    agent = AgentLoop(provider, config, memory)

    # Build a long conversation
    conversation = []
    for i in range(52):
        conversation.append({"role": "user", "content": f"msg {i}"})
        conversation.append({"role": "assistant", "content": f"reply {i}"})

    original_len = len(conversation)
    await agent._summarize_history(conversation)

    # Conversation should be preserved since summarization failed
    assert len(conversation) == original_len
    await agent.close()


# ── Token-aware compaction ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_aware_compaction_triggers(config, memory):
    """Compaction should trigger based on token budget, not message count."""
    # Use a small-tier model (effective context = 6000 tokens)
    # Build a conversation small enough by message count (<50) but big by tokens
    responses = [FakeResponse(content="done")]
    agent = _make_agent(config, memory, responses)

    # Each message ~500 chars = ~125 tokens. Need >4500 tokens (75% of 6000)
    # That's ~36 messages of 500 chars each → well under old 50-message limit
    conversation = []
    for i in range(20):
        conversation.append({"role": "user", "content": f"question {i} " + "x" * 480})
        conversation.append({"role": "assistant", "content": f"answer {i} " + "y" * 480})

    # The summarize call needs to succeed
    agent.provider.chat = AsyncMock(
        side_effect=[
            FakeResponse(content="Summary of conversation."),  # summarization
        ]
    )
    # Manually trigger what run() does after persisting
    from agent_mini.agent.token_estimator import estimate_messages_tokens
    current_tokens = estimate_messages_tokens(conversation)
    assert current_tokens > int(6000 * 0.75), "Conversation should exceed budget"
    assert len(conversation) < 50, "Should be under old message-count trigger"

    await agent._summarize_history(conversation)
    # After summarization, conversation should be shorter
    assert len(conversation) < 40
    await agent.close()


# ── Pruning ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prune_tool_results(config, memory):
    """Old tool results should be trimmed, recent ones preserved."""
    agent = _make_agent(config, memory, [])

    messages = [{"role": "system", "content": "sys"}]
    # Add many assistant + tool rounds
    for i in range(10):
        messages.append({"role": "assistant", "content": f"thinking {i}", "tool_calls": []})
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "name": "read_file",
            "content": f"{'X' * 6000}",  # big tool result
        })
    messages.append({"role": "user", "content": "now what?"})

    pruned = agent._prune_tool_results(messages)
    # Recent tool results (near the end) should be preserved or soft-trimmed
    # Very old ones should be placeholders
    old_tool = next(m for m in pruned if m.get("tool_call_id") == "tc0")
    assert len(old_tool["content"]) < 6000  # should be trimmed or replaced
    await agent.close()


# ── Max iterations by tier ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_iterations_uses_tier_default(tmp_path, memory):
    """When maxIterations is not set, tier default should be used."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    config = {
        "workspace": str(ws),
        "tools": {"restrictToWorkspace": False},
        "agent": {},  # No maxIterations set
    }
    agent = _make_agent(config, memory, [])
    # qwen2.5:7b → small tier → 15 iterations
    assert agent.max_iterations == 15
    await agent.close()


@pytest.mark.asyncio
async def test_max_iterations_respects_user_config(config, memory):
    """Explicit maxIterations in config should override tier default."""
    config["agent"]["maxIterations"] = 42
    agent = _make_agent(config, memory, [])
    assert agent.max_iterations == 42
    await agent.close()


# ── Finish reason (drives CLI exit codes) ───────────────────────────


async def test_finish_reason_text(config, memory):
    agent = _make_agent(config, memory, [FakeResponse(content="done")])
    await agent.run("hi", [])
    assert agent.finish_reason == "text"
    await agent.close()


async def test_finish_reason_max_iterations(config, memory):
    config["agent"]["maxIterations"] = 1
    call = FakeToolCall(id="tc1", name="list_directory", arguments={"path": "."})
    agent = _make_agent(config, memory, [FakeResponse(tool_calls=[call])])
    await agent.run("loop", [])
    assert agent.finish_reason == "max_iterations"
    await agent.close()


async def test_finish_reason_provider_error(config, memory):
    agent = _make_agent(config, memory, [ValueError("model not found")])
    result = await agent.run("hi", [])
    assert result.startswith("Error communicating with LLM")
    assert agent.finish_reason == "provider_error"
    await agent.close()


# ── Temperature (B6) ────────────────────────────────────────────────


async def test_null_temperature_is_passed_through(config, memory):
    config["agent"]["temperature"] = None
    agent = _make_agent(config, memory, [FakeResponse(content="ok")])
    await agent.run("hi", [])
    assert agent.provider.chat.call_args.kwargs["temperature"] is None
    await agent.close()


async def test_tier_override_changes_budgets(config, memory):
    config["agent"] = {"tier": "cloud"}
    agent = _make_agent(config, memory, [])
    assert agent.profile.tier == "cloud"
    assert agent.max_iterations == 25
    await agent.close()


# ── Model switching and detection (B2, B16, F7) ─────────────────────


async def test_num_ctx_is_requested_from_the_provider(config, memory):
    agent = _make_agent(config, memory, [])
    expected = num_ctx_for(agent.profile, agent.tools.get_tool_defs())
    assert agent.provider.context_window == expected
    assert expected % 2048 == 0 and expected > agent.profile.context
    await agent.close()


async def test_set_provider_recomputes_budgets(memory, tmp_path):
    ws = tmp_path / "ws"
    agent = _make_agent({"workspace": str(ws)}, memory, [])
    old = agent.provider
    new = AsyncMock()
    new.model_name = "llama3.1:70b"
    await agent.set_provider(new)
    assert agent.profile.tier == "large"
    assert agent.max_iterations == 25
    assert new.context_window == num_ctx_for(agent.profile, agent.tools.get_tool_defs())
    old.close.assert_awaited_once()
    await agent.close()


async def test_detect_model_uses_reported_size_and_context(memory, tmp_path):
    agent = _make_agent({"workspace": str(tmp_path / "ws")}, memory, [])
    agent.provider.model_info = AsyncMock(
        return_value=ModelInfo(params_b=0.6, context_length=4096, capabilities=["tools"])
    )
    await agent.detect_model()
    assert agent.profile.tier == "tiny"
    assert agent.provider.context_window == 4096
    assert agent.profile.context < 4096
    await agent.close()


# ── Text tool calls (F4) ────────────────────────────────────────────


async def test_tool_call_written_as_text_is_executed(config, memory):
    (Path(config["workspace"]) / "a.txt").write_text("from disk")
    responses = [
        FakeResponse(content='<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'),
        FakeResponse(content="done"),
    ]
    agent = _make_agent(config, memory, responses)
    assert await agent.run("read a.txt", []) == "done"
    second = agent.provider.chat.call_args_list[1][0][0]
    assert any(m["role"] == "tool" and m["content"] == "from disk" for m in second)
    assert agent.text_tool_calls == 1
    await agent.close()


# ── Loop detection and graceful stops (B13, B14, B15) ───────────────


async def test_repeated_calls_nudge_then_stop(config, memory):
    config["agent"]["maxIterations"] = 20
    call = FakeResponse(tool_calls=[FakeToolCall(id="tc", name="list_directory", arguments={})])
    agent = _make_agent(config, memory, [call] * 8 + [FakeResponse(content="I kept listing files.")])
    result = await agent.run("loop", [])
    assert agent.finish_reason == "stuck"
    assert result.startswith("I kept listing files.")
    fifth_call = agent.provider.chat.call_args_list[4][0][0]
    assert "repeated the same tool call" in fifth_call[-1]["content"]
    # The final answer is requested without tools.
    assert agent.provider.chat.call_args_list[-1].kwargs["tools"] is None
    await agent.close()


async def test_max_iterations_returns_a_summary(config, memory):
    config["agent"]["maxIterations"] = 2
    ws = Path(config["workspace"])
    (ws / "a.txt").write_text("a")
    (ws / "b.txt").write_text("b")
    responses = [
        FakeResponse(tool_calls=[FakeToolCall(id="1", name="read_file", arguments={"path": "a.txt"})]),
        FakeResponse(tool_calls=[FakeToolCall(id="2", name="read_file", arguments={"path": "b.txt"})]),
        FakeResponse(content="Read both files; still need to compare them."),
    ]
    agent = _make_agent(config, memory, responses)
    conversation: list[dict] = []
    result = await agent.run("compare", conversation)
    assert result.startswith("Read both files")
    assert agent.finish_reason == "max_iterations"
    assert "[tools used: read_file(a.txt), read_file(b.txt)]" in conversation[1]["content"]
    await agent.close()


async def test_provider_error_records_the_turn(config, memory):
    agent = _make_agent(config, memory, [ValueError("boom")])
    conversation: list[dict] = []
    await agent.run("hi", conversation)
    assert [m["role"] for m in conversation] == ["user", "assistant"]
    await agent.close()


async def test_no_retry_after_text_was_streamed(config, memory):
    agent = _make_agent(config, memory, [])

    async def flaky(messages, on_delta, **kwargs):
        await on_delta("partial")
        raise httpx.ConnectError("dropped")

    agent.provider.chat_stream = AsyncMock(side_effect=flaky)
    with patch("agent_mini.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await agent.run("hi", [], on_stream=AsyncMock())
    assert result.startswith("Error communicating with LLM")
    assert agent.provider.chat_stream.await_count == 1
    await agent.close()


async def test_thinking_callback_reaches_the_provider(config, memory):
    agent = _make_agent(config, memory, [FakeResponse(content="ok")])
    on_thinking = AsyncMock()
    await agent.run("hi", [], on_stream=AsyncMock(), on_thinking=on_thinking)
    assert agent.provider.chat_stream.call_args.kwargs["on_thinking"] is on_thinking
    await agent.close()


async def test_tool_trace_saved_with_the_reply(config, memory):
    (Path(config["workspace"]) / "t.txt").write_text("x")
    responses = [
        FakeResponse(tool_calls=[FakeToolCall(id="1", name="read_file", arguments={"path": "t.txt"})]),
        FakeResponse(content="It says x."),
    ]
    agent = _make_agent(config, memory, responses)
    conversation: list[dict] = []
    assert await agent.run("what's in t.txt?", conversation) == "It says x."
    assert conversation[1]["content"] == "It says x.\n\n[tools used: read_file(t.txt)]"
    await agent.close()


# ── Context budget (B11, B12) ───────────────────────────────────────


async def test_prune_stubs_oldest_results_to_fit_budget(config, memory):
    agent = _make_agent(config, memory, [])
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
        messages.append({"role": "tool", "tool_call_id": f"t{i}", "name": "read_file", "content": "Y" * 8000})

    pruned = agent._prune_tool_results(messages)
    assert pruned[2]["content"].startswith("[cleared to save context: read_file")
    assert pruned[-1]["content"] == "Y" * 8000  # the latest round is never cleared
    assert estimate_messages_tokens(pruned) <= agent.profile.context
    assert messages[2]["content"] == "Y" * 8000  # input untouched
    await agent.close()


async def test_prune_trims_old_results_even_below_the_old_threshold(memory, tmp_path):
    """Tiny tier caps output at 2000 chars; older results must still shrink."""
    ws = tmp_path / "ws"
    agent = _make_agent({"workspace": str(ws), "agent": {"tier": "tiny"}}, memory, [])
    messages = []
    for i in range(4):
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
        messages.append({"role": "tool", "tool_call_id": f"t{i}", "name": "x", "content": "Z" * 1900})
    pruned = agent._prune_tool_results(messages)
    assert len(pruned[1]["content"]) < 1900
    await agent.close()


async def test_summarize_keeps_the_latest_exchange(config, memory):
    agent = _make_agent(config, memory, [FakeResponse(content="summary")])
    conversation = [
        {"role": "user", "content": "write a long essay"},
        {"role": "assistant", "content": "w" * 40000},
    ]
    await agent._summarize_history(conversation)
    assert conversation[1]["content"] == "w" * 40000
    agent.provider.chat.assert_not_called()
    await agent.close()


async def test_summarize_starts_kept_history_on_a_user_turn(config, memory):
    agent = _make_agent(config, memory, [FakeResponse(content="summary")])
    conversation = [{"role": "user", "content": "[Previous context summary]\n" + "s" * 900}]
    for i in range(12):
        conversation.append({"role": "assistant" if i % 2 == 0 else "user", "content": f"m{i} " + "x" * 2000})
    await agent._summarize_history(conversation)
    assert conversation[0]["content"] == "[Previous context summary]\nsummary"
    assert conversation[1]["role"] == "user"
    request = agent.provider.chat.call_args[0][0][1]["content"]
    assert "s" * 900 in request  # the earlier summary is carried over whole
    await agent.close()
