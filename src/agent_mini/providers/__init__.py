"""LLM providers and the factory that picks one from the config."""

from __future__ import annotations

import os

from .base import ChatResponse, ModelInfo, Provider, ToolCall
from .ollama import OllamaProvider
from .openai import OpenAIProvider


def create_provider(config: dict) -> Provider:
    """The provider named by ``config["provider"]`` (a config that went through ``with_defaults``)."""
    name = config["provider"]
    cfg = config["providers"][name]
    if name == "ollama":
        return OllamaProvider(cfg["model"], cfg["baseUrl"], cfg["think"], cfg["numCtx"], cfg["keepAlive"])
    # Env vars are fallbacks so secrets don't have to live in config.json.
    key = cfg["apiKey"] or os.environ.get("AGENT_MINI_API_KEY") or (
        os.environ.get("OPENAI_API_KEY", "") if name == "openai" else ""
    )
    return OpenAIProvider(cfg["model"], cfg["baseUrl"], key, cfg["reasoningEffort"], name=name)


__all__ = ["ChatResponse", "ModelInfo", "OllamaProvider", "OpenAIProvider", "Provider", "ToolCall", "create_provider"]
