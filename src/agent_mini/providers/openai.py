"""OpenAI provider — thin wrapper over the OpenAI-compatible local provider."""

from __future__ import annotations

from .local import LocalProvider


class OpenAIProvider(LocalProvider):
    """OpenAI API (chat completions endpoint).

    Reuses ``LocalProvider`` since OpenAI *is* the reference implementation
    of the OpenAI chat completions protocol.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
    ):
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
        )

    @property
    def name(self) -> str:
        return "openai"
