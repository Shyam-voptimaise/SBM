from datetime import datetime

from capture.capture import (
    build_capture_metadata,
    build_coil_folder_name,
    build_coil_no,
    build_image_filename,
)
from capture.models import CameraConfig, CaptureConfig


def test_coil_folder_and_image_filename_preserve_legacy_format():
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    coil_no = build_coil_no(1)

    assert coil_no == "COIL_1"
    assert build_coil_folder_name(coil_no, timestamp) == (
        "COIL_20260102_030405_COIL_1"
    )
    assert build_image_filename(coil_no, "CAM1", "CAP1", timestamp) == (
        "COIL_1_CAM1_CAP1_030405.bmp"
    )


def test_capture_metadata_preserves_required_fields():
    started_at = datetime(2026, 1, 2, 3, 4, 5)
    captured_at = datetime(2026, 1, 2, 3, 4, 6)
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
        "COIL_1",
        "COIL_20260102_030405_COIL_1",
        started_at,
        camera,
        capture,
        captured_at,
    )

    assert metadata == {
        "coil_no": "COIL_1",
        "coil_folder": "COIL_20260102_030405_COIL_1",
        "coil_started_at": "2026-01-02T03:04:05",
        "coil_date": "2026-01-02",
        "camera_name": "CAM1",
        "camera_device_index": 0,
        "camera_serial_number": None,
        "capture_name": "CAP1",
        "delay_after_previous": 10,
        "captured_at": "2026-01-02T03:04:06",
    }
