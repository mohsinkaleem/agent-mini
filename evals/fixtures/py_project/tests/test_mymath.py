"""Tests for mymath — must still pass after the rename."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mymath import calc, double_calc  # noqa: E402


def test_calc_adds():
    assert calc(2, 3) == 5


def test_double_uses_calc():
    assert double_calc(4) == 8
