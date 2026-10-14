"""AgentLoop: tool rounds, retries, stops, budgets and history."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from agent_mini.loop import _SUMMARY_MARKER, AgentLoop
from agent_mini.prompt import estimate_messages_tokens, num_ctx_for
from agent_mini.providers.base import ChatResponse, ModelInfo, ToolCall
from tests.conftest import fake_provider


def _agent(config, memory, responses, model="qwen2.5:7b") -> AgentLoop:
    return AgentLoop(fake_provider(responses, model), config, memory)


def _call(name, **arguments) -> ChatResponse:
    return ChatResponse(tool_calls=[ToolCall(id=f"id-{name}", name=name, arguments=arguments)])


def _sent(agent, n) -> list[dict]:
    """Messages of the n-th provider call."""
    return agent.provider.chat.call_args_list[n][0][0]


async def test_text_answer(config, memory):
    agent = _agent(config, memory, [ChatResponse(content="Hello!")])
    conversation = []
    assert await agent.run("hi", conversation) == "Hello!"
    assert conversation == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    assert agent.finish_reason == "text"


async def test_tool_call_then_answer(config, memory, workspace):
    (workspace / "test.txt").write_text("file contents")
    agent = _agent(config, memory, [_call("read_file", path="test.txt"), ChatResponse(content="It says hi.")])
    assert await agent.run("read test.txt", []) == "It says hi."
    tool_msg = [m for m in _sent(agent, 1) if m["role"] == "tool"][0]
    assert tool_msg["content"] == "file contents"


async def test_parallel_tool_calls(config, memory, workspace):
    (workspace / "a.txt").write_text("aaa")
    (workspace / "b.txt").write_text("bbb")
    both = ChatResponse(tool_calls=[
        ToolCall(id="1", name="read_file", arguments={"path": "a.txt"}),
        ToolCall(id="2", name="read_file", arguments={"path": "b.txt"}),
    ])
    agent = _agent(config, memory, [both, ChatResponse(content="Got both")])
    assert await agent.run("read both", []) == "Got both"
    assert [m["content"] for m in _sent(agent, 1) if m["role"] == "tool"] == ["aaa", "bbb"]


async def test_tool_errors_reach_the_model(config, memory):
    agent = _agent(config, memory, [_call("read_file", path="missing.txt"), ChatResponse(content="Not there.")])
    await agent.run("read it", [])
    tool_msg = [m for m in _sent(agent, 1) if m["role"] == "tool"][0]
    assert tool_msg["content"] == "Error: missing.txt does not exist."


async def test_legacy_tool_names_still_work(config, memory, workspace):
    (workspace / "a.txt").write_text("x")
    agent = _agent(config, memory, [_call("list_directory"), ChatResponse(content="done")])
    await agent.run("ls", [])
    assert [m["content"] for m in _sent(agent, 1) if m["role"] == "tool"] == ["a.txt"]


async def test_max_iterations_saves_a_summary(config, memory, workspace):
    config["agent"]["maxIterations"] = 2
    (workspace / "a.txt").write_text("a")
    (workspace / "b.txt").write_text("b")
    responses = [_call("read_file", path="a.txt"), _call("read_file", path="b.txt"),
                 ChatResponse(content="Read both files; still need to compare them.")]
    agent = _agent(config, memory, responses)
    conversation: list[dict] = []
    result = await agent.run("compare", conversation)
    assert result.startswith("Read both files")
    assert "max iterations (2)" in result
    assert agent.finish_reason == "max_iterations"
    assert conversation[0] == {"role": "user", "content": "compare"}
    assert "[tools used: read_file(a.txt), read_file(b.txt)]" in conversation[1]["content"]
    assert agent.provider.chat.call_args_list[-1].kwargs["tools"] is None  # summary without tools


async def test_repeated_calls_nudge_then_stop(config, memory):
    config["agent"]["maxIterations"] = 20
    agent = _agent(config, memory, [_call("list_files")] * 8 + [ChatResponse(content="I kept listing files.")])
    result = await agent.run("loop", [])
    assert agent.finish_reason == "stuck"
    assert result.startswith("I kept listing files.")
    assert "repeated the same tool calls" in _sent(agent, 4)[-1]["content"]
    assert agent.provider.chat.call_args_list[-1].kwargs["tools"] is None


async def test_empty_reply_after_tools_gets_one_nudge(config, memory):
    responses = [_call("list_files"), ChatResponse(content=""), ChatResponse(content="Nothing here.")]
    agent = _agent(config, memory, responses)
    assert await agent.run("what's here?", []) == "Nothing here."
    assert "reply was empty" in _sent(agent, 2)[-1]["content"]


async def test_tool_call_written_as_text_is_executed(config, memory, workspace):
    (workspace / "a.txt").write_text("from disk")
    responses = [
        ChatResponse(content='<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'),
        ChatResponse(content="done"),
    ]
    agent = _agent(config, memory, responses)
    assert await agent.run("read a.txt", []) == "done"
    assert any(m["role"] == "tool" and m["content"] == "from disk" for m in _sent(agent, 1))
    assert agent.last_run["text_tool_calls"] == 1


async def test_last_run_report(config, memory):
    agent = _agent(config, memory, [_call("read_file", path="nope"), ChatResponse(
        content="done", usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})])
    await agent.run("go", [])
    run = agent.last_run
    assert run["finish_reason"] == "text" and run["reply"] == "done" and run["iterations"] == 2
    assert run["tool_calls"] == [{"name": "read_file", "args": {"path": "nope"}, "error": True}]
    assert run["usage"]["prompt_tokens"] == 10


async def test_tool_trace_saved_with_the_reply(config, memory, workspace):
    (workspace / "t.txt").write_text("x")
    agent = _agent(config, memory, [_call("read_file", path="t.txt"), ChatResponse(content="It says x.")])
    conversation: list[dict] = []
    await agent.run("what's in t.txt?", conversation)
    assert conversation[1]["content"] == "It says x.\n\n[tools used: read_file(t.txt)]"


async def test_long_results_are_clipped_with_a_hint(config, memory, workspace):
    (workspace / "big.txt").write_text("line\n" * 5000)
    agent = _agent(config, memory, [_call("read_file", path="big.txt"), ChatResponse(content="ok")])
    await agent.run("read", [])
    tool_msg = [m for m in _sent(agent, 1) if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) < agent.profile.output_limit + 200
    assert "Use offset/limit" in tool_msg["content"]


# ── Provider errors and retries ──────────────────────────────────────


async def test_retry_on_transient_error(config, memory):
    response = MagicMock(status_code=503)
    error = httpx.HTTPStatusError("unavailable", request=MagicMock(), response=response)
    agent = _agent(config, memory, [error, ChatResponse(content="Recovered!")])
    with patch("agent_mini.loop.asyncio.sleep", new_callable=AsyncMock):
        assert await agent.run("test", []) == "Recovered!"
    assert agent.provider.chat.call_count == 2


async def test_provider_error_records_the_turn(config, memory):
    agent = _agent(config, memory, [ValueError("model not found")])
    conversation: list[dict] = []
    result = await agent.run("hi", conversation)
    assert result.startswith("Error communicating with LLM")
    assert agent.finish_reason == "provider_error"
    assert [m["role"] for m in conversation] == ["user", "assistant"]


async def test_no_retry_after_text_was_streamed(config, memory):
    agent = _agent(config, memory, [])

    async def flaky(messages, on_delta=None, **kwargs):
        await on_delta("partial")
        raise httpx.ConnectError("dropped")

    agent.provider.chat = AsyncMock(side_effect=flaky)
    with patch("agent_mini.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await agent.run("hi", [], on_stream=AsyncMock())
    assert result.startswith("Error communicating with LLM")
    assert agent.provider.chat.await_count == 1


async def test_streaming_and_thinking_callbacks_reach_the_provider(config, memory):
    agent = _agent(config, memory, [ChatResponse(content="ok")])
    on_thinking = AsyncMock()
    await agent.run("hi", [], on_stream=AsyncMock(), on_thinking=on_thinking)
    kwargs = agent.provider.chat.call_args.kwargs
    assert kwargs["on_thinking"] is on_thinking and kwargs["on_delta"] is not None


# ── Profiles and budgets ─────────────────────────────────────────────


async def test_tier_defaults(workspace, memory):
    agent = _agent({"workspace": str(workspace)}, memory, [])
    assert agent.profile.tier == "small"
    assert agent.max_iterations == 15
    assert agent.temperature == agent.profile.temperature


async def test_config_overrides_the_tier(config, memory):
    config["agent"].update(maxIterations=42, temperature=0.1)
    agent = _agent(config, memory, [ChatResponse(content="ok")])
    assert agent.max_iterations == 42
    await agent.run("hi", [])
    assert agent.provider.chat.call_args.kwargs["temperature"] == 0.1


async def test_tier_override_changes_budgets(config, memory):
    config["agent"] = {"tier": "cloud"}
    agent = _agent(config, memory, [])
    assert agent.profile.tier == "cloud" and agent.max_iterations == 25


async def test_num_ctx_is_requested_from_the_provider(config, memory):
    agent = _agent(config, memory, [])
    expected = num_ctx_for(agent.profile, agent.tools.definitions())
    assert agent.provider.context_window == expected
    assert expected % 2048 == 0 and expected > agent.profile.context


async def test_set_provider_recomputes_budgets(workspace, memory):
    agent = _agent({"workspace": str(workspace)}, memory, [])
    old, new = agent.provider, fake_provider([], model="llama3.1:70b")
    await agent.set_provider(new)
    assert agent.profile.tier == "large" and agent.max_iterations == 25
    assert new.context_window == num_ctx_for(agent.profile, agent.tools.definitions())
    old.close.assert_awaited_once()


async def test_detect_model_uses_reported_size_and_context(workspace, memory):
    agent = _agent({"workspace": str(workspace)}, memory, [])
    agent.provider.model_info = AsyncMock(return_value=ModelInfo(0.6, 4096, ["tools"]))
    await agent.detect_model()
    assert agent.profile.tier == "tiny"
    assert agent.provider.context_window == 4096
    assert agent.profile.context < 4096


async def test_memories_reach_the_prompt_only_when_memory_is_on(workspace, memory):
    memory.store("editor", "helix")
    on = _agent({"workspace": str(workspace)}, memory, [ChatResponse(content="ok")])
    await on.run("hi", [])
    assert "editor: helix" in _sent(on, 0)[0]["content"]
    off = _agent({"workspace": str(workspace), "memory": {"enabled": False}}, memory, [ChatResponse(content="ok")])
    await off.run("hi", [])
    assert "helix" not in _sent(off, 0)[0]["content"]


# ── Context pruning and history summaries ────────────────────────────


async def test_prune_trims_old_tool_results(config, memory):
    agent = _agent(config, memory, [])
    messages = [{"role": "system", "content": "sys"}]
    for i in range(10):
        messages += [{"role": "assistant", "content": "", "tool_calls": []},
                     {"role": "tool", "tool_call_id": f"t{i}", "name": "read_file", "content": "X" * 6000}]
    pruned = agent._prune(messages)
    assert len(pruned[2]["content"]) < 6000
    assert messages[2]["content"] == "X" * 6000  # input untouched


async def test_prune_stubs_oldest_results_to_fit_budget(config, memory):
    agent = _agent(config, memory, [])
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages += [{"role": "assistant", "content": "", "tool_calls": []},
                     {"role": "tool", "tool_call_id": f"t{i}", "name": "read_file", "content": "Y" * 8000}]
    pruned = agent._prune(messages)
    assert pruned[2]["content"].startswith("[cleared to save context: read_file")
    assert pruned[-1]["content"] == "Y" * 8000  # the latest round is never cleared
    assert estimate_messages_tokens(pruned) <= agent.profile.context


async def test_prune_trims_small_results_for_tiny_models(workspace, memory):
    agent = _agent({"workspace": str(workspace), "agent": {"tier": "tiny"}}, memory, [])
    messages = []
    for i in range(4):
        messages += [{"role": "assistant", "content": "", "tool_calls": []},
                     {"role": "tool", "tool_call_id": f"t{i}", "name": "x", "content": "Z" * 1900}]
    assert len(agent._prune(messages)[1]["content"]) < 1900


async def test_long_history_is_summarized_after_the_turn(config, memory):
    conversation = []
    for i in range(20):
        conversation += [{"role": "user", "content": f"question {i} " + "x" * 480},
                         {"role": "assistant", "content": f"answer {i} " + "y" * 480}]
    agent = _agent(config, memory, [ChatResponse(content="ok"), ChatResponse(content="Summary.")])
    await agent.run("one more", conversation)
    assert conversation[0]["content"] == f"{_SUMMARY_MARKER}\nSummary."
    assert conversation[-2:] == [{"role": "user", "content": "one more"}, {"role": "assistant", "content": "ok"}]
    assert estimate_messages_tokens(conversation) < agent.profile.context * 0.75


async def test_failed_summary_keeps_the_history(config, memory):
    agent = _agent(config, memory, [Exception("summarization failed")])
    conversation = [{"role": r, "content": "m" * 400} for _ in range(52) for r in ("user", "assistant")]
    await agent._summarize_history(conversation)
    assert len(conversation) == 104


async def test_summary_keeps_the_latest_exchange(config, memory):
    agent = _agent(config, memory, [ChatResponse(content="summary")])
    conversation = [{"role": "user", "content": "write an essay"}, {"role": "assistant", "content": "w" * 40000}]
    await agent._summarize_history(conversation)
    assert conversation[1]["content"] == "w" * 40000
    agent.provider.chat.assert_not_called()


async def test_summary_starts_kept_history_on_a_user_turn(config, memory):
    agent = _agent(config, memory, [ChatResponse(content="summary")])
    conversation = [{"role": "user", "content": f"{_SUMMARY_MARKER}\n" + "s" * 900}]
    for i in range(12):
        conversation.append({"role": "assistant" if i % 2 == 0 else "user", "content": f"m{i} " + "x" * 2000})
    await agent._summarize_history(conversation)
    assert conversation[0]["content"] == f"{_SUMMARY_MARKER}\nsummary"
    assert conversation[1]["role"] == "user"
    assert "s" * 900 in agent.provider.chat.call_args[0][0][1]["content"]  # earlier summary carried over whole


def test_workspace_is_created(tmp_path: Path, memory):
    ws = tmp_path / "new" / "ws"
    AgentLoop(fake_provider([]), {"workspace": str(ws)}, memory)
    assert ws.is_dir()
