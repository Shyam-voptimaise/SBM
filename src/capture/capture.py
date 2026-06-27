import re
from datetime import datetime
from typing import Any, Dict, List, Sequence

from capture.models import CameraConfig, CaptureConfig, ScheduledCapture
from capture.time_utils import ist_date, to_ist


def build_coil_no(coil_counter: int) -> str:
    return f"{coil_counter:02d}"


def build_coil_folder_name(coil_no: str, started_at: datetime) -> str:
    ist_started_at = to_ist(started_at)
    return f"COIL_{ist_started_at.strftime('%Y%m%d_%H%M%S')}_{coil_no}"


def build_image_filename(
    coil_no: str,
    camera_name: str,
    capture_name: str,
) -> str:
    return (
        f"cam_{extract_number_token(camera_name)}_"
        f"cap_{extract_number_token(capture_name)}_"
        f"coil_{coil_no}.bmp"
    )


def extract_number_token(value: str) -> str:
    match = re.search(r"\d+", value)
    if match:
        return f"{int(match.group(0)):02d}"
    return _safe_text_token(value)


def build_capture_schedule(
    cameras: Sequence[CameraConfig],
) -> List[ScheduledCapture]:
    schedule = []

    for camera in cameras:
        capture_at = 0
        for capture in camera.captures:
            capture_at += capture.delay_after_previous_seconds
            schedule.append(
                ScheduledCapture(
                    capture_at_seconds=capture_at,
                    camera=camera,
                    capture=capture,
                )
            )

    return sorted(schedule, key=lambda item: item.capture_at_seconds)


def build_capture_metadata(
    coil_no: str,
    coil_folder: str,
    coil_started_at: datetime,
    camera: CameraConfig,
    capture: CaptureConfig,
    captured_at: datetime,
) -> Dict[str, Any]:
    coil_started_at = to_ist(coil_started_at)
    captured_at = to_ist(captured_at)

    return {
        "coil_no": coil_no,
        "coil_folder": coil_folder,
        "coil_started_at": coil_started_at.isoformat(),
        "coil_date": ist_date(coil_started_at),
        "camera_name": camera.name,
        "camera_device_index": camera.device_index,
        "camera_serial_number": camera.serial_number,
        "capture_name": capture.name,
        "delay_after_previous": capture.delay_after_previous_seconds,
        "captured_at": captured_at.isoformat(),
    }


def _safe_text_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", value).upper()
    return token or "00"
