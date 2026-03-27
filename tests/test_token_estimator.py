"""Tests for token estimation, model tier classification, and context budgets."""

import pytest

from agent_mini.agent.token_estimator import (
    classify_model_tier,
    estimate_messages_tokens,
    estimate_tokens,
    get_effective_context,
    get_output_limit,
    get_tier_max_iterations,
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
            ("unknown-model", "small"),  # default
        ],
    )
    def test_tiers(self, model, expected):
        assert classify_model_tier(model) == expected


class TestGetEffectiveContext:
    def test_small(self):
        assert get_effective_context("qwen2.5:7b") == 6000

    def test_cloud(self):
        assert get_effective_context("gemini-2.0-flash") == 32000

    def test_medium(self):
        assert get_effective_context("qwen2.5:14b") == 12000


class TestGetTierMaxIterations:
    def test_small_model(self):
        assert get_tier_max_iterations("llama3.1:8b") == 15

    def test_cloud_model(self):
        assert get_tier_max_iterations("gpt-4o-mini") == 25


class TestGetOutputLimit:
    def test_small(self):
        assert get_output_limit("qwen2.5:7b") == 4000

    def test_cloud(self):
        assert get_output_limit("gemini-2.0-flash") == 50000
