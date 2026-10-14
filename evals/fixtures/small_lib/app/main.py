from app.models import Event
from app.utils.dates import parse_date
from app.utils.strings import title_case


def main() -> None:
    event = Event(title_case("team sync"), parse_date("2026-03-02"))
    print(event)


if __name__ == "__main__":
    main()
