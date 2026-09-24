"""Tests for the Telegram channel's access control (no network)."""

from agent_mini.channels.telegram import TelegramChannel


def test_empty_allow_list_is_public():
    assert TelegramChannel("token").is_public
    assert TelegramChannel("token", ["*"]).is_public


def test_allow_list_restricts_users():
    ch = TelegramChannel("token", ["123", "alice"])
    assert not ch.is_public
    assert ch._is_allowed("123")
    assert ch._is_allowed("999", "alice")
    assert not ch._is_allowed("456", "mallory")
