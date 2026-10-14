"""Image references in user messages."""

from pathlib import Path

import agent_mini.vision as vision

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


def test_text_without_images():
    assert vision.image_parts("just text, and a file.json") is None


def test_local_images_and_urls(tmp_path: Path):
    (tmp_path / "shot.png").write_bytes(_PNG)
    parts = vision.image_parts("compare shot.png with https://example.com/a.png?x=1", workspace=tmp_path)
    assert parts[0] == {"type": "text", "text": "compare with"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2]["image_url"]["url"] == "https://example.com/a.png?x=1"


def test_local_images_can_be_disabled(tmp_path: Path):
    (tmp_path / "private.png").write_bytes(_PNG)
    assert vision.image_parts(f"describe {tmp_path / 'private.png'}", allow_local=False) is None
    parts = vision.image_parts("see https://example.com/a.png", allow_local=False)
    assert parts[-1]["image_url"]["url"] == "https://example.com/a.png"


def test_oversized_images_are_ignored(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 10)
    (tmp_path / "big.png").write_bytes(_PNG)
    assert vision.image_parts(f"describe {tmp_path / 'big.png'}") is None
