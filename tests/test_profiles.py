from datetime import datetime

from capture.models import CameraProfile
from capture.profiles import next_profile_change_at, select_active_profile
from capture.time_utils import IST


PROFILES = (
    CameraProfile(
        name="day",
        exposure_time=500000.0,
        gain_value=10.0,
        start_minutes=6 * 60,
    ),
    CameraProfile(
        name="night",
        exposure_time=700000.0,
        gain_value=12.0,
        start_minutes=18 * 60,
    ),
)


def test_select_active_profile_uses_latest_started_profile():
    active = select_active_profile(
        PROFILES,
        datetime(2026, 1, 2, 12, 0, tzinfo=IST),
    )

    assert active.name == "day"


def test_select_active_profile_wraps_across_midnight():
    active = select_active_profile(
        PROFILES,
        datetime(2026, 1, 2, 2, 0, tzinfo=IST),
    )

    assert active.name == "night"


def test_select_active_profile_switches_at_start_time():
    active = select_active_profile(
        PROFILES,
        datetime(2026, 1, 2, 18, 0, tzinfo=IST),
    )

    assert active.name == "night"


def test_next_profile_change_uses_next_start_today():
    change_at = next_profile_change_at(
        PROFILES,
        datetime(2026, 1, 2, 12, 0, tzinfo=IST),
    )

    assert change_at == datetime(2026, 1, 2, 18, 0, tzinfo=IST)


def test_next_profile_change_wraps_to_next_day():
    change_at = next_profile_change_at(
        PROFILES,
        datetime(2026, 1, 2, 19, 0, tzinfo=IST),
    )

    assert change_at == datetime(2026, 1, 3, 6, 0, tzinfo=IST)


def test_next_profile_change_returns_none_for_single_profile():
    change_at = next_profile_change_at(
        (PROFILES[0],),
        datetime(2026, 1, 2, 12, 0, tzinfo=IST),
    )

    assert change_at is None
