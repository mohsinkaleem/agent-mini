# Learning from Agent Mini

This guide uses Agent Mini to teach how an AI agent works. You start with a complete agent in about 100 lines, see what goes wrong with it on real models, and then read how the real agent deals with each problem.

**What you need**

- Python you can read comfortably: functions, dicts, classes. Part 3 uses `async`/`await`; Parts 1 and 2 don't.
- A model. Ollama with `qwen3.5:4b` (about 3 GB, fine on a laptop with 8 GB of RAM) or `qwen3.5:9b`, or a key for any OpenAI-compatible API.
- The code:

```bash
git clone https://github.com/mohsinkaleem/agent-mini.git && cd agent-mini
uv sync --extra dev          # or: pip install -e ".[dev]"
ollama pull qwen3.5:4b
```

**The plan**

1. [An agent in 100 lines](#part-1-an-agent-in-100-lines): run it and read it.
2. [What breaks, and what the real agent adds](#part-2-what-breaks-and-what-the-real-agent-adds)
3. [Reading the real code](#part-3-reading-the-real-code): an order to read it in, and what to skip.
4. [Watching it work](#part-4-watching-it-work): see every message the model gets.
5. [Exercises](#part-5-exercises)

---

## Part 1: An agent in 100 lines

[`examples/minimal_agent.py`](../examples/minimal_agent.py) is a whole agent. Run it:

```bash
python examples/minimal_agent.py "How many Python files are in the src folder?"
```

```
  → shell({'command': 'find src -name "*.py" | wc -l'})
  Run `find src -name "*.py" | wc -l`? [y/N] y
There are 21 Python files in the src folder.
```

### The one idea

A language model can't do anything; it can only write text. An agent is a loop around it:

1. Send the model the conversation, plus a list of tools it may ask for.
2. If it replies with text, that's the answer. Stop.
3. If it replies with tool calls, run them, add the results to the conversation, and go back to 1.

The conversation is a plain list of messages. After the run above it looks like this:

```
system     You are an assistant on the user's computer. Use the tools …
user       How many Python files are in the src folder?
assistant  tool_calls: [shell({"command": "find src -name \"*.py\" | wc -l"})]
tool       21
assistant  There are 21 Python files in the src folder.
```

Every call to the model sends the whole list. The model has no memory of its own; this list is its memory.

### The four parts of the file

1. **Tools** are ordinary Python functions that take strings and return a string.
2. **Tool schemas** describe each tool in JSON: its name, what it does, its arguments. The model never sees your Python, only these schemas. A tool with a vague description gets used badly.
3. **`ask_model`** is one HTTP request to `/chat/completions`. The reply is either text or a list of `tool_calls`.
4. **`run_agent`** is the loop.

### Things to notice

- **The model never runs anything.** It writes JSON asking for a tool, and your code decides whether and how to run it. That's why `shell` asks you first: the model chose that command, not you.
- **Errors are results.** When a tool fails, the error goes back to the model as text, and it usually tries something else. An agent that crashes on the first error is much less useful.
- **`tool_call_id`** ties each result to the call that asked for it. The model can ask for several tools at once.
- **`MAX_STEPS`** stops a model that never finishes.

Try a question that needs no tools ("What is a fork bomb?"), one that needs a file ("What does README.md say about plugins?") and one that needs the shell. Decline a command and see what the model does next.

---

## Part 2: What breaks, and what the real agent adds

The minimal agent works on a strong model and an easy task. Point it at a small model, a long task or a big file and it falls over. Each row below is a real failure, and the place in Agent Mini that handles it:

| In the minimal agent | What goes wrong | Where Agent Mini handles it |
|---|---|---|
| One failed HTTP request crashes it | Servers hiccup and rate-limit | `AgentLoop._call()` retries, in [loop.py](../src/agent_mini/loop.py) |
| The conversation grows forever | Small models get worse long before their context window is full | [context.py](../src/agent_mini/context.py): clip results, prune old ones, summarize old turns |
| Tool results are cut at 4,000 characters | A model can't tell what was cut or how to get the rest | `clip()` keeps head and tail and suggests `offset`/`limit` or `grep` |
| The model repeats the same call until `MAX_STEPS` | Small models get stuck | `_is_looping()` in loop.py: one nudge, then a forced answer |
| Hitting `MAX_STEPS` returns "Stopped" | You get nothing for all that work | `_final_answer()`: one last call without tools, asking for a summary |
| The model sends `file_path` instead of `path` | `TypeError`, and the model is confused | [forgiving.py](../src/agent_mini/forgiving.py) maps common names |
| `read_file("/etc/passwd")` just works | The agent can read anything you can | [sandbox.py](../src/agent_mini/sandbox.py): paths stay in the workspace |
| You approve every shell command or none | Approving everything gets tedious; approving nothing is risky | `tools.confirm` per tool, a blocklist, secrets kept out of the shell's environment |
| It can read files but not change them | Real tasks need edits | `edit_file` in [tools.py](../src/agent_mini/tools.py), with `/undo` |
| Every run starts from scratch | | [memory.py](../src/agent_mini/memory.py) and [sessions.py](../src/agent_mini/sessions.py) |
| One prompt and one budget for every model | A 4B and a 70B model need different limits | Model tiers in [prompt.py](../src/agent_mini/prompt.py) |

See it for yourself: run the minimal agent with a tiny model, then Agent Mini with the same model and question.

```bash
MINI_MODEL=qwen3.5:0.8b python examples/minimal_agent.py "What version is in pyproject.toml?"
agent-mini chat --model qwen3.5:0.8b --workspace . -m "What version is in pyproject.toml?"
```

### Small-model fixes

Many of Agent Mini's extra lines exist because small models (4B to 9B parameters) make predictable mistakes. Each of those workarounds is marked with a comment:

```bash
grep -rn "Small-model fix" src/
```

Skip them on a first read. The agent's logic is the same without them; they make it work on a model that runs on a laptop.

---

## Part 3: Reading the real code

Read the files in this order. The line counts are approximate.

| # | File | Lines | Read it for | Skip at first |
|---|---|---|---|---|
| 1 | [providers/base.py](../src/agent_mini/providers/base.py) | 110 | `ToolCall`, `ChatResponse` and `Provider.chat()`: the types everything else uses | `raise_for_status` |
| 2 | [loop.py](../src/agent_mini/loop.py) | 360 | `run()`: its numbered steps 1–4 are the minimal agent's loop | how `_call()` retries |
| 3 | [prompt.py](../src/agent_mini/prompt.py) | 180 | The tier table and the system prompt's rules | the model-name regexes |
| 4 | [tools.py](../src/agent_mini/tools.py) | 470 | `@tool` and `_schema()` (compare with the minimal agent's `schema()`), then the steps of `execute()`, then `read_file`, `write_file`, `shell` | `list_files`, `search_files` |
| 5 | [context.py](../src/agent_mini/context.py) | 110 | `clip`, `prune`, `summarize_history` | |
| 6 | [sandbox.py](../src/agent_mini/sandbox.py) | 130 | Paths, the command blocklist, killing a process group | the `rm -rf` regex |
| 7 | [providers/parsing.py](../src/agent_mini/providers/parsing.py), [forgiving.py](../src/agent_mini/forgiving.py), [editing.py](../src/agent_mini/editing.py) | 310 | The small-model fixes, once you know what they fix | |
| 8 | [providers/ollama.py](../src/agent_mini/providers/ollama.py), [providers/openai.py](../src/agent_mini/providers/openai.py) | 240 | Two wire formats, and streaming | |

Leave these for later: [cli.py](../src/agent_mini/cli.py) is about 670 lines, mostly terminal UI. [web.py](../src/agent_mini/web.py) mostly parses HTML, and [config.py](../src/agent_mini/config.py) holds settings and defaults.

Questions to answer as you read:

- `run()` sends `self._prune(turn.messages)` to the model, not `turn.messages`. Why a pruned copy instead of pruning in place?
- After a turn, only your message and the final reply are saved to the history; the tool calls are dropped. What does the `[tools used: …]` note at the end of the reply make up for?
- `execute()` never raises. What would happen to the loop if it did?
- `write_file` refuses to replace a file the agent hasn't read. What did small models do that made this necessary? (The comment says.)
- Why is the summary of old turns a `user` message and not a second `system` message?

[TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) explains the reasons for each design choice, file by file.

---

## Part 4: Watching it work

**Every message of a run.** `--report` writes the whole conversation the model saw, including tool calls and results:

```bash
mkdir -p /tmp/demo && echo "eggs" > /tmp/demo/todo.txt
agent-mini chat --workspace /tmp/demo --yes --report /tmp/run.json -m 'Add "buy milk" to todo.txt'
```

```bash
python - <<'EOF'
import json
for m in json.load(open("/tmp/run.json"))["messages"]:
    calls = [c["function"]["name"] + c["function"]["arguments"] for c in m.get("tool_calls") or []]
    print(m["role"].ljust(10), repr(str(m.get("content"))[:70]), *calls)
EOF
```

The report also holds the finish reason, each tool call (with an error flag), the number of iterations and the token usage.

**Logs as it runs.** `-v` prints each tool call with its arguments and the start of its result.

**The interactive tour.** [how-it-works.html](how-it-works.html) steps through one message, from your prompt to the saved turn, with each step's messages. It also has a tool-call checker you can type into. Open it in a browser.

---

## Part 5: Exercises

Run `uv run pytest` after each change to the real agent.

### With the minimal agent

1. **Add a `write_file` tool.** Write the function, then its schema. `schema()` only takes one argument, so extend it or write that schema out by hand. Ask the agent to create `hello.txt`. *Then:* ask it to add a line to an existing file. What does it write? (Compare with `write_file` in tools.py.)
2. **Show what the model says while it works.** Models sometimes write a sentence alongside their tool calls. Print it.
3. **Count tokens.** `ask_model` returns only the message, but the full reply also has `usage`. Print the prompt tokens on each step and watch them grow. What's taking up most of them?

### With the real agent

4. **Write a plugin.** Make a `word_count(path)` tool as a plugin (the plugin example is under *Tools* in the [README](../README.md#tools)), then check `/tools` in `agent-mini chat`.
5. **Add a built-in tool.** Add `file_info(path)`, returning size and modification time, as a `@tool` method in tools.py, with a test in `tests/test_tools.py`. Every tool's schema is sent with every request: how many tokens did yours add? (`/status` shows usage.)
6. **Catch longer loops.** `_is_looping()` in loop.py catches `A A A A` and `A B A B`. Make it catch `A B C A B C` too. Write the test in `tests/test_loop.py` first. Why does it compare results as well as calls? Think of a task that rightly makes the same call twice.
7. **Tighten the budget.** Set `"agent": {"contextWindow": 2000}` in `~/.agent-mini/config.json`, then give the agent a task that reads several files, with `--report`. Search the report for `[cleared to save context`: which results went, and did the model cope? Remove the setting afterwards.

### With the evals

8. **Write an eval task.** Copy `evals/tasks/edit_setting.toml`, change the setup, prompt and checks, and add a `solution`. Run `uv run python evals/run.py --validate`: your checks must fail before the solution and pass after it. Then run it on a model with `--task <your id>`.
9. **Test a change the way the project does.** Pick one rule in the system prompt (`_RULES` in prompt.py), change or remove it, and compare before and after:

    ```bash
    uv run python evals/run.py --model qwen3.5:4b --repeat 3 --label before
    # make your change
    uv run python evals/run.py --model qwen3.5:4b --repeat 3 --label after
    uv run python evals/run.py --compare evals/results/<before>.json evals/results/<after>.json
    ```

    Did it change what you expected? Did it change anything you didn't expect?

### Stretch

10. **Memory by meaning.** `recall()` in memory.py matches keywords. Replace it with embeddings from Ollama (`ollama pull nomic-embed-text`, then `POST /api/embed`) and cosine similarity. What do you gain, and what does each `remember` now cost?

---

## Words used here

| Word | Meaning |
|---|---|
| Agent | A program that runs a model in a loop and lets it use tools |
| Tool call | The model's request to run a tool: a name and JSON arguments |
| Tool schema | The JSON description of a tool that the model sees |
| ReAct loop | The pattern in `run()`: the model reasons, acts with a tool, sees the result, repeats |
| Token | The unit models read and write; roughly 4 characters of English |
| Context window | The most tokens a model can take in at once |
| Context budget | The limit Agent Mini keeps the conversation under. It is set per tier, well below the context window, where the model still works well |
| Tier | Agent Mini's size class for a model (tiny, small, medium, large, cloud), which sets its budgets |
| Sandbox | The limits on what tools may touch |
| Prompt injection | Text in a file or web page that tries to give the model instructions |
| Eval | A task with automatic checks, run against a real model to measure the agent |
