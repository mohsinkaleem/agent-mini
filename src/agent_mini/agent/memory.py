"""Persistent memory — simple JSON-backed key/value store with fuzzy search."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

log = logging.getLogger("agent-mini")

# ---------- Minimal Porter-style stemmer (covers common suffixes) ----------

_SUFFIX_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("isation", "ize"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
    ("biliti", "ble"), ("ling", "l"),
    ("ing", ""), ("ment", ""), ("ness", ""), ("ity", ""),
    ("ies", "y"), ("es", "e"), ("ed", ""), ("ly", ""), ("s", ""),
]


def _stem(word: str) -> str:
    """Very lightweight suffix stripping — handles 80 %+ of common English inflections."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, and stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if len(w) > 1]


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
            except json.JSONDecodeError:
                log.warning("Corrupted memory file %s, starting fresh", self.filepath)
                self._data = []
            except OSError as e:
                log.error("Cannot read memory file %s: %s", self.filepath, e)
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
        """TF-IDF fuzzy search across stored memories."""
        if not self._data:
            return "No matching memories found."

        query_tokens = _tokenize(query)
        if not query_tokens:
            return "No matching memories found."

        # Build document token lists
        docs: list[tuple[dict, list[str]]] = []
        for entry in self._data:
            text = f"{entry['key']} {entry['value']}"
            tokens = _tokenize(text)
            if tokens:
                docs.append((entry, tokens))

        if not docs:
            return "No matching memories found."

        # IDF: log(N / df) for each term in the query
        n = len(docs)
        df: Counter = Counter()
        for _, tokens in docs:
            unique = set(tokens)
            for t in unique:
                df[t] += 1

        idf = {t: math.log((n + 1) / (df.get(t, 0) + 1)) + 1 for t in query_tokens}

        # Score each document
        scored: list[tuple[float, dict]] = []
        for entry, tokens in docs:
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for qt in query_tokens:
                # TF-IDF with length normalization
                score += (tf.get(qt, 0) / doc_len) * idf.get(qt, 0)
                # Substring fallback: partial match bonus
                for token in set(tokens):
                    if qt in token or token in qt:
                        score += 0.3 * idf.get(qt, 0) / doc_len
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
