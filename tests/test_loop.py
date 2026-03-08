"""Tests for AgentLoop — parallel execution, self-reflection, retry."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_mini.agent.loop import AgentLoop
from agent_mini.agent.memory import Memory


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class FakeResponse:
    content: str = ""
    tool_calls: list = field(default_factory=list)


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
    result = await agent.run("read it", conversation)

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
    assert "maximum iterations" in result.lower()
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
