"""Telegram gateway: access control, routing and startup policy (no network)."""

import asyncio

from agent_mini.config import with_defaults
from agent_mini.telegram import TelegramGateway, chunk_text, gateway_config


class _SlowAgent:
    def __init__(self):
        self.active = self.peak = 0

    async def run(self, text, session, on_stream=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        session.append({"role": "user", "content": text})
        self.active -= 1
        return f"re: {text}"


def test_access_control():
    assert TelegramGateway(None, "t", []).is_public
    assert TelegramGateway(None, "t", ["*"]).is_public
    gw = TelegramGateway(None, "t", ["123", "alice"])
    assert not gw.is_public
    assert gw.is_allowed("123") and gw.is_allowed("999", "alice") and not gw.is_allowed("456", "mallory")


async def test_one_message_at_a_time_per_user():
    agent = _SlowAgent()
    gw = TelegramGateway(agent, "t", ["1", "2"])
    await asyncio.gather(*(gw.handle("1", f"m{i}") for i in range(3)))
    assert agent.peak == 1 and len(gw.sessions["1"]) == 3
    await asyncio.gather(gw.handle("1", "a"), gw.handle("2", "b"))
    assert agent.peak == 2 and gw.sessions["2"] == [{"role": "user", "content": "b"}]


def _telegram(**tg):
    return with_defaults({"channels": {"telegram": {"enabled": True, **tg}}})


def test_public_bot_runs_readonly_without_memory():
    config, warnings = gateway_config(_telegram())
    assert config["tools"]["sandboxLevel"] == "readonly" and not config["memory"]["enabled"]
    assert "PUBLIC" in warnings[0]
    config, warnings = gateway_config(_telegram(allowShell=True))
    assert config["tools"]["sandboxLevel"] == "workspace" and "shell is enabled" in warnings[0]


def test_private_bot_keeps_the_config():
    config, warnings = gateway_config(_telegram(allowFrom=["123"]))
    assert config["tools"]["sandboxLevel"] == "workspace" and config["memory"]["enabled"] and warnings == []
    config, warnings = gateway_config(_telegram(allowFrom=["123", "alice"]))
    assert not config["memory"]["enabled"] and "usernames" in warnings[0]
    config, _ = gateway_config(_telegram(allowFrom=["1", "2"], sharedMemory=True))
    assert config["memory"]["enabled"]


def test_chunk_text():
    assert chunk_text("") == ["(empty response)"]
    assert [len(c) for c in chunk_text("x" * 9000)] == [4000, 4000, 1000]
