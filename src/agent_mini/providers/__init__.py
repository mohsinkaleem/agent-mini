"""LLM Provider registry — single entry-point for creating providers."""

from __future__ import annotations

import os

from .base import BaseProvider, ChatResponse, ToolCall
from .local import LocalProvider
from .ollama import OllamaProvider


def create_provider(config: dict) -> BaseProvider:
    """Instantiate the active provider based on *config*."""
    provider_name = config.get("provider", "ollama")
    providers_cfg = config.get("providers", {})
    cfg = providers_cfg.get(provider_name, {})

    # Env vars are fallbacks so secrets don't have to live in config.json.
    env_key = os.environ.get("AGENT_MINI_API_KEY")

    match provider_name:
        case "ollama":
            return OllamaProvider(
                base_url=cfg.get("baseUrl", "http://localhost:11434"),
                model=cfg.get("model", "llama3.1"),
                think=cfg.get("think"),
                num_ctx=cfg.get("numCtx"),
                keep_alive=cfg.get("keepAlive"),
            )
        case "openai":
            # OpenAI is the reference implementation of the protocol LocalProvider speaks.
            return LocalProvider(
                base_url="https://api.openai.com/v1",
                api_key=cfg.get("apiKey") or env_key or os.environ.get("OPENAI_API_KEY", ""),
                model=cfg.get("model", "gpt-4o"),
                reasoning_effort=cfg.get("reasoningEffort"),
                name="openai",
            )
        case "local":
            return LocalProvider(
                base_url=cfg.get("baseUrl", "http://localhost:8080/v1"),
                api_key=cfg.get("apiKey") or env_key or "no-key",
                model=cfg.get("model", "local-model"),
                reasoning_effort=cfg.get("reasoningEffort"),
            )
        case _:
            raise ValueError(
                f"Unknown provider: {provider_name!r}. "
                f"Available: ollama, openai, local"
            )


__all__ = [
    "BaseProvider",
    "ChatResponse",
    "ToolCall",
    "OllamaProvider",
    "LocalProvider",
    "create_provider",
]
