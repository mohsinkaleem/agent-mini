from pricing import apply_discount


class Cart:
    def __init__(self):
        self.items: list[tuple[str, float, int]] = []

    def add(self, name: str, price: float, qty: int = 1) -> None:
        self.items.append((name, price, qty))

    def total(self, discount_percent: float = 0) -> float:
        subtotal = sum(price * qty for _, price, qty in self.items)
        return apply_discount(subtotal, discount_percent)
