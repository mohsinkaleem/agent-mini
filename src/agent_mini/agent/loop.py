"""Core agent loop — think → act → observe → repeat."""

from __future__ import annotations

import json
import logging

from ..providers.base import BaseProvider, ChatResponse
from .context import build_system_prompt
from .memory import Memory
from .tools import ToolExecutor

log = logging.getLogger("agent-mini")


class AgentLoop:
    """Runs the agentic ReAct loop.

    1. Build messages (system + history + user message).
    2. Call the LLM.
    3. If the LLM returns tool calls → execute them, append results, goto 2.
    4. If the LLM returns text → return it to the caller.
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: dict,
        memory: Memory,
    ):
        self.provider = provider
        self.config = config
        self.memory = memory
        self.tools = ToolExecutor(config, memory)
        self.max_iterations: int = config.get("agent", {}).get("maxIterations", 20)
        self.temperature: float = config.get("agent", {}).get("temperature", 0.7)

    async def run(self, user_message: str, conversation: list[dict]) -> str:
        """Process *user_message* and return the assistant's final text reply.

        *conversation* is mutated in-place (appended with the new user/assistant
        turns) so the caller can maintain session state.
        """
        system_prompt = build_system_prompt(self.config, self.memory)
        tool_defs = self.tools.get_tool_defs()

        # Build the full message list for this request
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            log.debug("iteration %d / %d", iteration + 1, self.max_iterations)

            try:
                response: ChatResponse = await self.provider.chat(
                    messages,
                    tools=tool_defs or None,
                    temperature=self.temperature,
                )
            except Exception as e:
                log.error("Provider error: %s", e)
                return f"Error communicating with LLM: {e}"

            # ----- text-only response → done -----
            if not response.tool_calls:
                final = response.content or "(empty response)"
                # Persist to conversation history
                conversation.append({"role": "user", "content": user_message})
                conversation.append({"role": "assistant", "content": final})
                # Keep history bounded
                if len(conversation) > 50:
                    conversation[:] = conversation[-40:]
                return final

            # ----- tool calls → execute and continue -----
            assistant_msg: dict = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments
                            ),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                log.info(
                    "🔧 %s(%s)",
                    tc.name,
                    json.dumps(tc.arguments, ensure_ascii=False)[:200],
                )
                result = await self.tools.execute(tc.name, tc.arguments)
                log.debug("   → %s", result[:300])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )

        return "Reached maximum iterations. Please try a simpler request."
