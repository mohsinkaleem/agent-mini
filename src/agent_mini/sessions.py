"""Session persistence — save and resume conversations."""

from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path

from .config import agent_home, write_private

log = logging.getLogger("agent-mini")

_VALID_ID = re.compile(r"^[\w-]+$")


def _sessions_dir() -> Path:
    return agent_home() / "sessions"


def _ensure_dir() -> Path:
    d = _sessions_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _session_path(session_id: str) -> Path | None:
    """Path for *session_id*, or None if the ID could escape the sessions dir."""
    if not _VALID_ID.match(session_id):
        return None
    return _sessions_dir() / f"{session_id}.json"


def save_session(session_id: str, conversation: list[dict]) -> Path:
    """Persist a conversation to disk. Returns the file path."""
    path = _session_path(session_id)
    if path is None:
        raise ValueError(f"Invalid session ID: {session_id!r}")
    _ensure_dir()
    data = {
        "id": session_id,
        "updated": datetime.now().isoformat(),
        "conversation": conversation,
    }
    write_private(path, json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_session(session_id: str) -> list[dict] | None:
    """Load a conversation from disk. Returns None if not found."""
    path = _session_path(session_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("conversation", [])
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("Failed to load session %s: %s", session_id, e)
        return None


def list_sessions() -> list[dict]:
    """Return metadata for all saved sessions, newest first."""
    d = _ensure_dir()
    sessions = []
    for path in d.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            msg_count = len(data.get("conversation", []))
            sessions.append({
                "id": data.get("id", path.stem),
                "updated": data.get("updated", "?"),
                "messages": msg_count,
                "preview": _preview(data.get("conversation", [])),
            })
        except json.JSONDecodeError:
            log.warning("Corrupted session file: %s", path)
            continue
        except OSError as e:
            log.warning("Cannot read session %s: %s", path, e)
            continue
    sessions.sort(key=lambda s: s["updated"], reverse=True)
    return sessions


def generate_session_id() -> str:
    """Timestamp plus a random suffix, so two terminals never collide."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


def _preview(conversation: list[dict], max_len: int = 80) -> str:
    """Get a short preview of the conversation."""
    for msg in conversation:
        if msg.get("role") == "user" and msg.get("content"):
            text = msg["content"].replace("\n", " ")[:max_len]
            return text
    return "(empty)"
