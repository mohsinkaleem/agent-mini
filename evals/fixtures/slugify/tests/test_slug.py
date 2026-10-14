import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slug import slugify  # noqa: E402


def test_lowercase_and_dashes():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_is_dropped():
    assert slugify("Hello, World!") == "hello-world"


def test_whitespace_collapses():
    assert slugify("  many   spaces  ") == "many-spaces"


def test_no_double_dashes():
    assert slugify("a - b") == "a-b"


def test_digits_are_kept():
    assert slugify("Top 10 Tips") == "top-10-tips"
