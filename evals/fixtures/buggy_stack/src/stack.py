"""A buggy stack implementation. The `pop` method is off-by-one — the eval
asks the agent to find and fix the bug so the tests pass."""


class Stack:
    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        # BUG: returns the wrong end of the list
        return self._items.pop(0)

    def peek(self):
        return self._items[-1]

    def __len__(self):
        return len(self._items)
