"""Images in user messages: local paths and URLs become image content parts."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_IMAGE_URL = re.compile(r"https?://\S+\.(?:png|jpe?g|gif|webp|bmp)", re.IGNORECASE)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _local_image(ref: str, workspace: Path | None) -> Path | None:
    """*ref* as an existing image file of sane size (relative paths resolve against *workspace*)."""
    if Path(ref).suffix.lower() not in _IMAGE_EXTENSIONS:
        return None
    p = Path(ref).expanduser()
    if not p.is_absolute() and workspace is not None:
        p = workspace / p
    try:
        return p if p.is_file() and p.stat().st_size <= MAX_IMAGE_BYTES else None
    except OSError:
        return None


def image_parts(text: str, workspace: Path | None = None, allow_local: bool = True) -> list[dict] | None:
    """OpenAI-style content parts for a message that mentions images, or None if it mentions none.

    Local files are skipped when *allow_local* is False (remote chat users).
    """
    words, images = [], []
    for word in text.split():
        ref = word.strip("\"'(),;[]")
        if allow_local and (path := _local_image(ref, workspace)):
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            images.append(f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}")
        elif _IMAGE_URL.match(ref):
            images.append(ref)
        else:
            words.append(word)
    if not images:
        return None
    parts = [{"type": "text", "text": " ".join(words)}] if words else []
    return parts + [{"type": "image_url", "image_url": {"url": url}} for url in images]
