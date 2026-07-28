import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stack import Stack  # noqa: E402


def test_push_pop_is_lifo():
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.pop() == 3
    assert s.pop() == 2
    assert s.pop() == 1


def test_peek_does_not_mutate():
    s = Stack()
    s.push("a")
    s.push("b")
    assert s.peek() == "b"
    assert len(s) == 2
