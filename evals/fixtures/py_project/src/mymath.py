"""A tiny arithmetic helper — used by the eval fixture."""


def calc(a: int, b: int) -> int:
    """Add two ints. The eval will ask the agent to rename this."""
    return a + b


def double_calc(x: int) -> int:
    """Uses `calc` — the rename must update the call site too."""
    return calc(x, x)
