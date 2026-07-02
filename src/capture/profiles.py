from datetime import datetime, timedelta
from typing import Optional, Sequence

from capture.models import CameraProfile
from capture.time_utils import to_ist


def select_active_profile(
    profiles: Sequence[CameraProfile],
    at: datetime,
) -> CameraProfile:
    if not profiles:
        raise ValueError("at least one camera profile is required")

    current_time = to_ist(at).time()
    current_minutes = current_time.hour * 60 + current_time.minute
    ordered_profiles = sorted(profiles, key=lambda profile: profile.start_minutes)
    active_profile = ordered_profiles[-1]

    for profile in ordered_profiles:
        if profile.start_minutes <= current_minutes:
            active_profile = profile
        else:
            break

    return active_profile


def next_profile_change_at(
    profiles: Sequence[CameraProfile],
    at: datetime,
) -> Optional[datetime]:
    if not profiles:
        raise ValueError("at least one camera profile is required")

    start_minutes = sorted({profile.start_minutes for profile in profiles})
    if len(start_minutes) <= 1:
        return None

    current = to_ist(at)
    candidates = []

    for profile_start_minutes in start_minutes:
        hour, minute = divmod(profile_start_minutes, 60)
        candidate = current.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )
        if candidate <= current:
            candidate += timedelta(days=1)
        candidates.append(candidate)

    return min(candidates)
