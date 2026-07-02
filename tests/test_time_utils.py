from datetime import datetime

from capture.time_utils import IST, next_ist_midnight, seconds_until_next_ist_midnight


def test_next_ist_midnight_uses_following_ist_date():
    now = datetime(2026, 1, 2, 23, 30, tzinfo=IST)

    assert next_ist_midnight(now) == datetime(2026, 1, 3, 0, 0, tzinfo=IST)
    assert seconds_until_next_ist_midnight(now) == 30 * 60
