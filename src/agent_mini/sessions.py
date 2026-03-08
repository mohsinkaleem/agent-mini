"""Session persistence — save and resume conversations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("agent-mini")

_SESSIONS_DIR = Path.home() / ".agent-mini" / "sessions"


def _ensure_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def save_session(
    session_id: str,
    conversation: list[dict],
    metadata: dict | None = None,
) -> Path:
    """Persist a conversation to disk. Returns the file path."""
    d = _ensure_dir()
    path = d / f"{session_id}.json"
    data = {
        "id": session_id,
        "updated": datetime.now().isoformat(),
        "metadata": metadata or {},
        "conversation": conversation,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_session(session_id: str) -> list[dict] | None:
    """Load a conversation from disk. Returns None if not found."""
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
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


def delete_session(session_id: str) -> bool:
    """Delete a saved session. Returns True if deleted."""
    path = _SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def generate_session_id() -> str:
    """Generate a timestamp-based session ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _preview(conversation: list[dict], max_len: int = 80) -> str:
    """Get a short preview of the conversation."""
    for msg in conversation:
        if msg.get("role") == "user" and msg.get("content"):
            text = msg["content"].replace("\n", " ")[:max_len]
            return text
    return "(empty)"
