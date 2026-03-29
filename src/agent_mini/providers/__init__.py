"""LLM Provider registry — single entry-point for creating providers."""

from __future__ import annotations

from .base import BaseProvider, ChatResponse, ToolCall
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .github_copilot import GitHubCopilotProvider
from .local import LocalProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider


def create_provider(config: dict) -> BaseProvider:
    """Instantiate the active provider based on *config*."""
    provider_name = config.get("provider", "ollama")
    providers_cfg = config.get("providers", {})
    cfg = providers_cfg.get(provider_name, {})

    match provider_name:
        case "ollama":
            return OllamaProvider(
                base_url=cfg.get("baseUrl", "http://localhost:11434"),
                model=cfg.get("model", "llama3.1"),
                think=cfg.get("think"),
            )
        case "gemini":
            return GeminiProvider(
                api_key=cfg.get("apiKey", ""),
                model=cfg.get("model", "gemini-2.0-flash"),
            )
        case "openai":
            return OpenAIProvider(
                api_key=cfg.get("apiKey", ""),
                model=cfg.get("model", "gpt-4o"),
            )
        case "claude":
            return ClaudeProvider(
                api_key=cfg.get("apiKey", ""),
                model=cfg.get("model", "claude-sonnet-4-20250514"),
            )
        case "github_copilot":
            return GitHubCopilotProvider(
                token=cfg.get("token", ""),
                model=cfg.get("model", "gpt-4o"),
            )
        case "local":
            return LocalProvider(
                base_url=cfg.get("baseUrl", "http://localhost:8080/v1"),
                api_key=cfg.get("apiKey", "no-key"),
                model=cfg.get("model", "local-model"),
            )
        case _:
            raise ValueError(
                f"Unknown provider: {provider_name!r}. "
                f"Available: ollama, gemini, openai, claude, github_copilot, local"
            )


__all__ = [
    "BaseProvider",
    "ChatResponse",
    "ToolCall",
    "OllamaProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "GitHubCopilotProvider",
    "LocalProvider",
    "create_provider",
]
