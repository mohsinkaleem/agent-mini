def apply_discount(price: float, percent: float) -> float:
    """Price after a percentage discount (10 means 10% off), rounded to cents."""
    return round(price * percent / 100, 2)
