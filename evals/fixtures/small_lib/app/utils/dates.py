from datetime import date, datetime


def parse_date(text: str) -> date:
    """Parse an ISO date such as 2026-03-02."""
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5
