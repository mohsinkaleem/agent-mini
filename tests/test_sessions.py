"""Session storage hardening (B20, S9)."""

import pytest

import agent_mini.sessions as sess
from agent_mini.config import agent_home


def test_ids_do_not_collide():
    assert len({sess.generate_session_id() for _ in range(20)}) == 20


def test_path_traversal_ids_are_rejected():
    with pytest.raises(ValueError):
        sess.save_session("../escape", [])
    secret = agent_home() / "secret.json"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('{"conversation": [{"role": "user", "content": "x"}]}')
    assert sess.load_session("../secret") is None
    assert secret.exists()


def test_session_file_is_private():
    path = sess.save_session("s1", [{"role": "user", "content": "hi"}])
    assert path.stat().st_mode & 0o777 == 0o600
    assert sess.load_session("s1") == [{"role": "user", "content": "hi"}]
