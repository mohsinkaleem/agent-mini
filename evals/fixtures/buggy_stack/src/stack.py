"""A simple LIFO stack."""


class Stack:
    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop(0)

    def peek(self):
        return self._items[-1]

    def __len__(self):
        return len(self._items)
