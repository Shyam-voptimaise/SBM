from capture.capture import build_capture_schedule
from capture.models import CameraConfig, CaptureConfig


def test_capture_schedule_is_cumulative_and_sorted_across_cameras():
    cameras = (
        CameraConfig(
            name="CAM1",
            device_index=0,
            serial_number=None,
            exposure_time=500000.0,
            gain_value=10.0,
            captures=(
                CaptureConfig(name="CAP1", delay_after_previous_seconds=10),
                CaptureConfig(name="CAP2", delay_after_previous_seconds=6),
            ),
        ),
        CameraConfig(
            name="CAM2",
            device_index=1,
            serial_number=None,
            exposure_time=500000.0,
            gain_value=10.0,
            captures=(
                CaptureConfig(name="CAP1", delay_after_previous_seconds=10),
                CaptureConfig(name="CAP2", delay_after_previous_seconds=6),
            ),
        ),
    )

    schedule = build_capture_schedule(cameras)

    assert [
        (item.capture_at_seconds, item.camera.name, item.capture.name)
        for item in schedule
    ] == [
        (10, "CAM1", "CAP1"),
        (10, "CAM2", "CAP1"),
        (16, "CAM1", "CAP2"),
        (16, "CAM2", "CAP2"),
    ]
