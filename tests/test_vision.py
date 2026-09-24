"""Image references in user messages (S9)."""

from pathlib import Path

import agent_mini.agent.vision as vision

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


def test_relative_paths_resolve_against_workspace(tmp_path: Path):
    (tmp_path / "shot.png").write_bytes(_PNG)
    parts = vision.build_image_content_parts("what is in shot.png", workspace=tmp_path)
    assert parts and parts[-1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_local_images_can_be_disabled(tmp_path: Path):
    img = tmp_path / "private.png"
    img.write_bytes(_PNG)
    assert vision.build_image_content_parts(f"describe {img}", allow_local=False) is None
    # URLs still work for remote users.
    parts = vision.build_image_content_parts("see https://example.com/a.png", allow_local=False)
    assert parts[-1]["image_url"]["url"] == "https://example.com/a.png"


def test_oversized_images_are_ignored(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 10)
    img = tmp_path / "big.png"
    img.write_bytes(_PNG)
    assert vision.build_image_content_parts(f"describe {img}") is None
