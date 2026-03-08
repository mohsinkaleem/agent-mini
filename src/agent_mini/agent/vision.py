"""Vision utilities — image encoding for multi-modal providers."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_URL_PATTERN = re.compile(r"https?://\S+\.(?:png|jpe?g|gif|webp|bmp)", re.IGNORECASE)


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


def build_image_content_parts(text: str) -> list[dict] | None:
    """Parse message text for image references and build multi-part content.

    Returns None if no images are found (use plain text message instead).
    Supports:
    - Local file paths: /path/to/image.png or relative paths
    - Image URLs: https://example.com/image.jpg
    """
    parts: list[dict] = []
    remaining_text = text

    # Extract image file paths (words ending with image extensions)
    words = text.split()
    image_refs: list[str] = []
    text_parts: list[str] = []

    for word in words:
        clean = word.strip("\"'(),;[]")
        if is_image_path(clean) and Path(clean).expanduser().exists():
            image_refs.append(clean)
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
