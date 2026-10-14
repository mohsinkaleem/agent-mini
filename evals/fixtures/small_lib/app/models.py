from dataclasses import dataclass
from datetime import date


@dataclass
class Event:
    title: str
    day: date
