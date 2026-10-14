"""Persistent memory: a JSON list of key/value facts with keyword search."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from .config import write_private

log = logging.getLogger("agent-mini")


def _stems(text: str) -> set[str]:
    """Crude stems ("deploying", "deployment" → "deplo"): enough to rank a few hundred facts."""
    return {w.rstrip("s")[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class Memory:
    """Facts stored as ``{key, value, timestamp}`` in a JSON file. One entry per key."""

    def __init__(self, path: Path, max_entries: int = 1000):
        self.path = path
        self.max_entries = max_entries
        self._data: list[dict] = []
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Keep the bad file instead of overwriting it on the next save.
                backup = path.with_suffix(path.suffix + ".corrupt")
                path.replace(backup)
                log.warning("Corrupted memory file, moved to %s and started fresh", backup)
            except OSError as e:
                log.error("Cannot read memory file %s: %s", path, e)

    def _save(self) -> None:
        try:
            write_private(self.path, json.dumps(self._data, indent=2, ensure_ascii=False))
        except OSError as e:
            log.error("Cannot write memory file %s: %s", self.path, e)

    def _without(self, key: str) -> list[dict]:
        norm = key.strip().lower()
        return [e for e in self._data if e["key"].strip().lower() != norm]

    def store(self, key: str, value: str) -> str:
        """Save *value* under *key*, replacing an entry with the same key."""
        kept = self._without(key)
        replaced = len(kept) < len(self._data)
        kept.append({"key": key, "value": value, "timestamp": datetime.now().isoformat(timespec="seconds")})
        self._data = kept[-self.max_entries:]
        self._save()
        return f"{'Updated' if replaced else 'Stored'} memory: {key}"

    def forget(self, key: str) -> str:
        kept = self._without(key)
        if len(kept) == len(self._data):
            return f"No memory with key '{key}'."
        self._data = kept
        self._save()
        return f"Forgot memory: {key}"

    def recall(self, query: str, limit: int = 10) -> str:
        """Entries sharing the most word stems with *query*, newest first among equals."""
        wanted = _stems(query)
        scored = [
            (len(wanted & _stems(f"{e['key']} {e['value']}")), i, e) for i, e in enumerate(self._data)
        ]
        hits = sorted((s for s in scored if s[0]), key=lambda s: (s[0], s[1]), reverse=True)
        if not hits:
            return "No matching memories found."
        return "\n".join(f"{e['key']}: {e['value']}" for _, _, e in hits[:limit])

    def recent(self, n: int = 5) -> list[dict]:
        """The *n* newest entries (shown in the system prompt)."""
        return self._data[-n:] if n > 0 else []
