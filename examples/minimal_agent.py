"""A whole AI agent in about 100 lines: the idea behind agent-mini, without the extras.

An agent is a loop. Send the model the conversation and a list of tools. If it
answers in text, you're done. If it asks for tools, run them, add the results
to the conversation, and ask again.

Run it (needs only httpx, which agent-mini already installs):

    ollama pull qwen3.5:9b
    python examples/minimal_agent.py "How many Python files are in this folder?"

Any OpenAI-compatible server works:

    MINI_BASE_URL=https://api.openai.com/v1 MINI_MODEL=gpt-5-mini OPENAI_API_KEY=sk-... \
        python examples/minimal_agent.py "What's in README.md?"

docs/LEARN.md walks through this file, then shows where each part lives in the
real agent and what it adds.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

BASE_URL = os.environ.get("MINI_BASE_URL", "http://localhost:11434/v1")  # Ollama's OpenAI-compatible API
MODEL = os.environ.get("MINI_MODEL", "qwen3.5:9b")
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed-for-ollama")
MAX_STEPS = 10  # tool rounds before we give up
MAX_RESULT_CHARS = 4000  # a huge tool result would crowd everything else out of the model's context

SYSTEM_PROMPT = (
    "You are an assistant on the user's computer. Use the tools to look at files and run commands "
    "instead of guessing. When you are done, reply with a short answer."
)


# ── 1. Tools: plain Python functions that return text ────────────────
# Real version: src/agent_mini/tools.py


def shell(command: str) -> str:
    # The model chose this command, so a person approves it first.
    if input(f"  Run `{command}`? [y/N] ").strip().lower() != "y":
        return "The user declined to run this command."
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
    return (result.stdout + result.stderr).strip() or "(no output)"


def read_file(path: str) -> str:
    return Path(path).read_text(errors="replace")


TOOLS = {"shell": shell, "read_file": read_file}


# ── 2. Tool schemas: how the model learns which tools exist ──────────
# Real version: the @tool decorator in tools.py builds these from each function's signature.


def schema(name: str, description: str, param: str) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {param: {"type": "string"}}, "required": [param]},
    }}


TOOL_SCHEMAS = [
    schema("shell", "Run a shell command and return its output.", "command"),
    schema("read_file", "Read a text file.", "path"),
]


# ── 3. One call to the model ─────────────────────────────────────────
# Real version: src/agent_mini/providers/ (streaming, retries, Ollama's native API)


def ask_model(messages: list[dict]) -> dict:
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]  # {"role": "assistant", "content": ..., "tool_calls": ...}


# ── 4. The loop ──────────────────────────────────────────────────────
# Real version: AgentLoop.run() in src/agent_mini/loop.py


def run_tool(call: dict) -> str:
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
        print(f"  → {name}({args})")
        if name not in TOOLS:
            return f"Error: there is no tool called {name}."
        return TOOLS[name](**args)
    except Exception as e:
        # Errors go back to the model as text: it can read them and try something else.
        return f"Error: {e}"


def run_agent(task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_STEPS):
        reply = ask_model(messages)
        messages.append(reply)  # the model's turn is part of the conversation too
        if not reply.get("tool_calls"):
            return reply.get("content") or ""  # no tool calls: this is the answer
        for call in reply["tool_calls"]:
            result = run_tool(call)
            # tool_call_id ties the result to the call that asked for it.
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result[:MAX_RESULT_CHARS]})
    return f"Stopped after {MAX_STEPS} steps without an answer."


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or input("Task: ")
    print(run_agent(task))
