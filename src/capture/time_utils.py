from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def ist_date(value: datetime) -> str:
    return to_ist(value).strftime("%Y-%m-%d")


def next_ist_midnight(value: datetime) -> datetime:
    current = to_ist(value)
    next_date = current.date() + timedelta(days=1)
    return datetime.combine(next_date, time.min, tzinfo=IST)


def seconds_until_next_ist_midnight(value: datetime) -> float:
    current = to_ist(value)
    return max((next_ist_midnight(current) - current).total_seconds(), 0.0)
