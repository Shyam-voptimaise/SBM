from datetime import datetime
from typing import Any, Dict, List, Sequence

from capture.models import CameraConfig, CaptureConfig, ScheduledCapture


def build_coil_no(coil_counter: int) -> str:
    return f"COIL_{coil_counter}"


def build_coil_folder_name(coil_no: str, started_at: datetime) -> str:
    return f"COIL_{started_at.strftime('%Y%m%d_%H%M%S')}_{coil_no}"


def build_image_filename(
    coil_no: str,
    camera_name: str,
    capture_name: str,
    captured_at: datetime,
) -> str:
    return f"{coil_no}_{camera_name}_{capture_name}_{captured_at.strftime('%H%M%S')}.bmp"


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
    return {
        "coil_no": coil_no,
        "coil_folder": coil_folder,
        "coil_started_at": coil_started_at.isoformat(),
        "coil_date": coil_started_at.strftime("%Y-%m-%d"),
        "camera_name": camera.name,
        "camera_device_index": camera.device_index,
        "camera_serial_number": camera.serial_number,
        "capture_name": capture.name,
        "delay_after_previous": capture.delay_after_previous_seconds,
        "captured_at": captured_at.isoformat(),
    }
