import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cart import Cart  # noqa: E402


def test_total_without_discount():
    cart = Cart()
    cart.add("pen", 2.5, qty=4)
    assert cart.total() == 10.0


def test_total_with_discount():
    cart = Cart()
    cart.add("book", 20.0)
    cart.add("pen", 2.5, qty=4)
    assert cart.total(10) == 27.0
