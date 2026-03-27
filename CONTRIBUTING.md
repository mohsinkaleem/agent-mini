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

- **Stay lean.** The core is ~3,500 LOC. Don't add frameworks, heavy abstractions, or dependencies unless absolutely necessary.
- **Zero-framework.** Pure `httpx` + `asyncio`. No LangChain, no LiteLLM, no abstractions-on-abstractions.
- **One file, one job.** Each module has a clear, single responsibility.
- **Tests are required.** Every new feature or bug fix needs a test in `tests/`.

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
- New LLM provider integrations (following `providers/base.py` interface)
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

## Project Structure

```
src/agent_mini/
├── cli.py              # CLI entry point (Click)
├── config.py           # Typed config loading
├── bus.py              # Multi-channel message routing
├── sessions.py         # Session persistence
├── agent/
│   ├── loop.py         # Core ReAct agent loop
│   ├── context.py      # System prompt builder
│   ├── memory.py       # Persistent JSON memory + TF-IDF search
│   ├── tools.py        # Built-in tools + plugin loader
│   ├── token_estimator.py  # Token counting + model tier classification
│   └── vision.py       # Image detection + encoding
├── providers/
│   ├── base.py         # Provider interface + tool call parsing
│   ├── ollama.py       # Ollama provider
│   ├── gemini.py       # Google Gemini provider
│   ├── github_copilot.py  # GitHub Copilot (OAuth device flow)
│   └── local.py        # Any OpenAI-compatible endpoint
└── channels/
    ├── base.py         # Channel interface
    └── telegram.py     # Telegram bot channel
```

## Releasing

Releases are automated via GitHub Actions:

1. Update `version` in both `pyproject.toml` and `src/agent_mini/__init__.py`.
2. Commit: `git commit -am "release: v0.2.0"`
3. Tag: `git tag v0.2.0`
4. Push: `git push && git push --tags`
5. Create a GitHub Release from the tag — PyPI publish runs automatically.
