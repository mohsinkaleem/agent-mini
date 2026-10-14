"""A tiny arithmetic helper."""


def calc(a: int, b: int) -> int:
    """Add two ints."""
    return a + b


def double_calc(x: int) -> int:
    """Twice x, via calc."""
    return calc(x, x)
