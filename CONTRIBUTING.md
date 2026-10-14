# Contributing to Agent Mini

Thanks for your interest in contributing! Agent Mini is intentionally lean — every line of code must earn its place.

## Development Setup

```bash
# Clone
git clone https://github.com/mohsinkaleem/agent-mini.git
cd agent-mini

# Install with dev dependencies (uses uv)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## Project Principles

- **Stay lean.** The core is about 3,000 lines. Don't add frameworks, heavy abstractions, or dependencies unless absolutely necessary.
- **Zero-framework.** Pure `httpx` + `asyncio`. No LangChain, no LiteLLM, no abstractions-on-abstractions.
- **Flat and obvious.** One module per job, no subpackages beyond `providers/`.
- **Measure small-model changes.** A change to the prompt, the tools or the loop needs an eval comparison (`python evals/run.py --repeat 3`, before and after, `--compare` the two reports) on at least one small model.
- **Tests are required.** Every new feature or bug fix needs a test in `tests/`. A behaviour bug seen in an eval transcript should also get an eval task.

## How to Contribute

### Bug Reports

Open an issue with:
1. What you expected
2. What happened
3. Steps to reproduce
4. Your provider/model (e.g., Ollama + qwen2.5:7b)

### Pull Requests

1. Fork the repo and create a branch from `main`.
2. Make your changes — keep the diff small and focused.
3. Add or update tests in `tests/`.
4. Run the test suite: `uv run pytest tests/ -v`
5. Run the linter: `uv run ruff check src/ tests/`
6. Open a PR with a clear description of *what* and *why*.

### What We Welcome

- Bug fixes
- New LLM provider integrations (a `Provider` subclass in `providers/`)
- New built-in tools (keep them small and self-contained)
- Plugin examples
- Documentation improvements
- Performance improvements (with benchmarks)

### What We Avoid

- Adding large dependencies (if it can be done with stdlib, do it with stdlib)
- Framework integrations (LangChain, LlamaIndex, etc.)
- Features that increase complexity without clear user value
- Refactors that change everything but fix nothing

## Code Style

- Python 3.11+, type hints where they help readability.
- Ruff for linting (`ruff check`).
- No docstrings required for obvious methods — code should be self-explanatory.
- Prefer `match/case` for dispatch over `if/elif` chains.
- Line length 120 (`ruff check src tests evals`).

## Project Structure

```
src/agent_mini/
├── cli.py          # init, chat, doctor, gateway
├── config.py       # defaults (the one source of truth), validation, files
├── loop.py         # the ReAct loop
├── prompt.py       # model tiers, token budgets, system prompt
├── tools.py        # tool registry: one @tool method per tool, plus plugins
├── web.py          # web_search, web_fetch, SSRF guard
├── memory.py       # persistent memory
├── vision.py       # images in messages
├── sessions.py     # saved conversations
├── telegram.py     # Telegram gateway
└── providers/
    ├── base.py     # Provider interface, tool-call parsing and JSON repair
    ├── ollama.py   # Ollama native API
    └── openai.py   # OpenAI and any OpenAI-compatible server
evals/              # task-eval harness: run.py, tasks/*.toml, fixtures/
```

A new tool is a method on `Tools` decorated with `@tool(description, **param_docs)`; its signature becomes the JSON schema. A new setting goes into `DEFAULTS` in `config.py`, and `config.example.json` is regenerated from it (a test checks they match).

## Releasing

Releases are automated via GitHub Actions:

1. Update `version` in both `pyproject.toml` and `src/agent_mini/__init__.py`.
2. Commit: `git commit -am "release: v0.2.0"`
3. Tag: `git tag v0.2.0`
4. Push: `git push && git push --tags`
5. Create a GitHub Release from the tag — PyPI publish runs automatically.
