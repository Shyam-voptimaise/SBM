from datetime import datetime

from capture.capture import (
    build_capture_metadata,
    build_coil_folder_name,
    build_coil_no,
    build_image_filename,
)
from capture.models import CameraConfig, CaptureConfig
from capture.time_utils import IST


def test_coil_folder_and_image_filename_use_ist_and_sequence_format():
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=IST)
    coil_no = build_coil_no(1)

    assert coil_no == "01"
    assert build_coil_folder_name(coil_no, timestamp) == (
        "COIL_20260102_030405_01"
    )
    assert build_image_filename(coil_no, "CAM1", "CAP1") == (
        "cam_01_cap_01_coil_01.bmp"
    )


def test_capture_metadata_preserves_required_fields():
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=IST)
    captured_at = datetime(2026, 1, 2, 3, 4, 6, tzinfo=IST)
    camera = CameraConfig(
        name="CAM1",
        device_index=0,
        serial_number=None,
        exposure_time=500000.0,
        gain_value=10.0,
        captures=(),
    )
    capture = CaptureConfig(name="CAP1", delay_after_previous_seconds=10)

    metadata = build_capture_metadata(
        "01",
        "COIL_20260102_030405_01",
        started_at,
        camera,
        capture,
        captured_at,
    )

    assert metadata == {
        "coil_no": "01",
        "coil_folder": "COIL_20260102_030405_01",
        "coil_started_at": "2026-01-02T03:04:05+05:30",
        "coil_date": "2026-01-02",
        "camera_name": "CAM1",
        "camera_device_index": 0,
        "camera_serial_number": None,
        "capture_name": "CAP1",
        "delay_after_previous": 10,
        "captured_at": "2026-01-02T03:04:06+05:30",
    }
