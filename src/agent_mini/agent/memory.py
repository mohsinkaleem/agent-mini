"""Persistent memory — simple JSON-backed key/value store with search."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Memory:
    """Lightweight persistent memory.

    Stores entries as ``{key, value, timestamp}`` dicts in a JSON file.
    Recall uses simple keyword matching — no embedding model required.
    """

    def __init__(self, filepath: Path, max_entries: int = 1000):
        self.filepath = filepath
        self.max_entries = max_entries
        self._data: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.filepath.exists():
            try:
                self._data = json.loads(self.filepath.read_text())
            except (json.JSONDecodeError, Exception):
                self._data = []

    def _save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Public API (called by tools)
    # ------------------------------------------------------------------

    def store(self, key: str, value: str) -> str:
        """Store a key/value pair. Returns confirmation string."""
        entry = {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        self._data.append(entry)
        if len(self._data) > self.max_entries:
            self._data = self._data[-self.max_entries :]
        self._save()
        return f"Stored memory: {key}"

    def recall(self, query: str) -> str:
        """Keyword search across stored memories."""
        keywords = query.lower().split()
        scored: list[tuple[int, dict]] = []

        for entry in self._data:
            text = f"{entry['key']} {entry['value']}".lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return "No matching memories found."

        lines = []
        for _, entry in scored[:10]:
            lines.append(f"[{entry['timestamp']}] {entry['key']}: {entry['value']}")
        return "\n".join(lines)

    def get_recent(self, n: int = 5) -> list[dict]:
        """Return the *n* most recent entries (for system prompt context)."""
        return self._data[-n:]
