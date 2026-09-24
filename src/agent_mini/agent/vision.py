"""Vision utilities — image encoding for multi-modal providers."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_URL_PATTERN = re.compile(r"https?://\S+\.(?:png|jpe?g|gif|webp|bmp)", re.IGNORECASE)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def is_image_path(text: str) -> bool:
    """Check if text looks like an image file path."""
    return Path(text).suffix.lower() in _IMAGE_EXTENSIONS


def is_image_url(text: str) -> bool:
    """Check if text looks like an image URL."""
    return bool(_URL_PATTERN.match(text))


def encode_image_base64(path: str | Path) -> tuple[str, str]:
    """Read an image file and return (base64_data, mime_type)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return data, mime


def _local_image(ref: str, workspace: Path | None) -> Path | None:
    """Resolve *ref* (relative paths against *workspace*) to an existing image of sane size."""
    p = Path(ref).expanduser()
    if not p.is_absolute() and workspace is not None:
        p = workspace / p
    try:
        return p if p.is_file() and p.stat().st_size <= MAX_IMAGE_BYTES else None
    except OSError:
        return None


def build_image_content_parts(
    text: str,
    workspace: Path | None = None,
    allow_local: bool = True,
) -> list[dict] | None:
    """Parse message text for image references and build multi-part content.

    Returns None if no images are found (use plain text message instead).
    Supports:
    - Local file paths: /path/to/image.png, or relative to *workspace*
      (skipped when *allow_local* is False, e.g. for remote chat users;
      files over 10 MB are ignored)
    - Image URLs: https://example.com/image.jpg
    """
    parts: list[dict] = []

    # Extract image file paths (words ending with image extensions)
    words = text.split()
    image_refs: list[str] = []
    text_parts: list[str] = []

    for word in words:
        clean = word.strip("\"'(),;[]")
        local = _local_image(clean, workspace) if allow_local and is_image_path(clean) else None
        if local:
            image_refs.append(str(local))
        elif is_image_url(clean):
            image_refs.append(clean)
        else:
            text_parts.append(word)

    if not image_refs:
        return None

    # Add text part
    remaining = " ".join(text_parts).strip()
    if remaining:
        parts.append({"type": "text", "text": remaining})

    # Add image parts
    for ref in image_refs:
        if is_image_url(ref):
            parts.append({
                "type": "image_url",
                "image_url": {"url": ref},
            })
        else:
            data, mime = encode_image_base64(ref)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })

    return parts
