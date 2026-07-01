from datetime import datetime
from typing import Sequence

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
