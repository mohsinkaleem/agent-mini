"""Tests for token estimation, model tier classification, and context budgets."""

import pytest

from agent_mini.agent.token_estimator import (
    classify_model_tier,
    estimate_messages_tokens,
    estimate_tokens,
    get_profile,
)


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world!") == 3  # 12 chars / 4

    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_long(self):
        text = "a" * 4000
        assert estimate_tokens(text) == 1000


class TestEstimateMessagesTokens:
    def test_simple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        total = estimate_messages_tokens(msgs)
        assert total > 0
        # 4 overhead * 2 msgs + tokens for content
        assert total == 4 + 4 + 4 + 1  # 13

    def test_tool_calls_counted(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "test.txt"}',
                        },
                    }
                ],
            }
        ]
        total = estimate_messages_tokens(msgs)
        assert total > 4  # more than just overhead


class TestClassifyModelTier:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("llama3.2:3b", "tiny"),
            ("phi-4-mini", "tiny"),
            ("qwen2.5:1.5b", "tiny"),
            ("qwen2.5:7b", "small"),
            ("llama3.1:8b", "small"),
            ("gemma3:4b", "small"),
            ("mistral:7b", "small"),
            ("qwen2.5:14b", "medium"),
            ("phi-4:14b", "medium"),
            ("mistral-nemo:12b", "medium"),
            ("gemini-2.0-flash", "cloud"),
            ("gpt-4o-mini", "cloud"),
            ("gpt-4.1-mini", "cloud"),
            # 20-72B open-weight models are "large"; bigger ones count as cloud.
            ("llama3.1:70b", "large"),
            ("qwen2.5:32b", "large"),
            ("mixtral:8x7b", "large"),
            ("unknown-model", "small"),  # default
            # B1: current model names
            ("gpt-5", "cloud"),
            ("gpt-5-mini", "cloud"),
            ("o3", "cloud"),
            ("o4-mini", "cloud"),
            ("claude-sonnet-4", "cloud"),
            ("gpt-oss:20b", "large"),
            ("qwen3.5:27b", "large"),
            ("gemma3:27b", "large"),
            ("mistral-small:24b", "large"),
            ("qwen3:30b-a3b", "large"),
            ("gpt-oss:120b", "cloud"),
            ("llama3.1:405b", "cloud"),
            ("deepseek-r1:671b", "cloud"),
            ("mixtral:8x22b", "cloud"),
            ("qwen3:0.6b", "tiny"),
            ("smollm2:135m", "tiny"),
            ("qwen3:8b", "small"),
            ("hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "small"),
            ("llama3.1", "small"),
        ],
    )
    def test_tiers(self, model, expected):
        assert classify_model_tier(model) == expected


class TestGetProfile:
    def test_tier_override(self):
        assert get_profile("qwen2.5:7b", {"tier": "cloud"}).tier == "cloud"

    def test_context_window_override(self):
        profile = get_profile("qwen2.5:7b", {"contextWindow": 16000})
        assert profile.tier == "small"
        assert profile.context == 16000

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValueError, match="agent.tier"):
            get_profile("qwen2.5:7b", {"tier": "huge"})

    def test_large_between_medium_and_cloud(self):
        medium, large, cloud = (
            get_profile("qwen2.5:14b"), get_profile("qwen3.5:27b"), get_profile("gpt-5")
        )
        assert medium.context < large.context < cloud.context
        assert medium.output_limit < large.output_limit < cloud.output_limit


class TestProfileBudgets:
    def test_context(self):
        assert get_profile("qwen2.5:7b").context == 6000
        assert get_profile("qwen2.5:14b").context == 12000
        assert get_profile("gemini-2.0-flash").context == 32000

    def test_max_iterations(self):
        assert get_profile("llama3.1:8b").max_iterations == 15
        assert get_profile("gpt-4o-mini").max_iterations == 25

    def test_output_limit(self):
        assert get_profile("qwen2.5:7b").output_limit == 4000
        assert get_profile("gemini-2.0-flash").output_limit == 50000
