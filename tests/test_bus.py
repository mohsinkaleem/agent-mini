"""Gateway message routing (S7)."""

import asyncio

from agent_mini.bus import MessageBus


class _SlowAgent:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def run(self, text, session, on_stream=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        session.append({"role": "user", "content": text})
        self.active -= 1
        return f"re: {text}"


async def test_same_user_messages_run_one_at_a_time():
    agent = _SlowAgent()
    bus = MessageBus(agent)
    await asyncio.gather(*(bus.handle_message("tg", "1", f"m{i}") for i in range(3)))
    assert agent.peak == 1
    assert len(bus.sessions["tg:1"]) == 3


async def test_different_users_run_concurrently_with_separate_history():
    agent = _SlowAgent()
    bus = MessageBus(agent)
    await asyncio.gather(bus.handle_message("tg", "1", "a"), bus.handle_message("tg", "2", "b"))
    assert agent.peak == 2
    assert bus.sessions["tg:1"] == [{"role": "user", "content": "a"}]
