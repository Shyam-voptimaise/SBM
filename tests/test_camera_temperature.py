import queue
import time

from capture.camera import (
    BaslerCameraManager,
    format_temperature_reading,
    read_camera_temperature,
)
from capture.models import (
    CameraConfig,
    CameraRuntimeConfig,
    CameraTemperatureReading,
    GPIOConfig,
    LoggingConfig,
    PathsConfig,
    RuntimeConfig,
    UploadConfig,
)
from capture.runtime import CaptureRuntime


class FakeTemperature:
    def __init__(self, value):
        self.Value = value


class FakeCamera:
    def __init__(self, temperature):
        self.TemperatureAbs = FakeTemperature(temperature)


class BrokenTemperature:
    @property
    def Value(self):
        raise RuntimeError("sensor missing")


class BrokenTemperatureCamera:
    TemperatureAbs = BrokenTemperature()


def camera_config(name="CAM1", device_index=0):
    return CameraConfig(
        name=name,
        device_index=device_index,
        serial_number=None,
        captures=(),
    )


def test_read_camera_temperature_uses_basler_temperature_abs_value():
    assert read_camera_temperature(FakeCamera(37.26)) == 37.26


def test_collect_temperature_readings_formats_detected_camera(tmp_path):
    manager = BaslerCameraManager(
        (camera_config(),),
        tmp_path,
        queue.Queue(),
    )
    manager.cameras["CAM1"] = FakeCamera(37.26)

    readings = manager.collect_temperature_readings(should_reconnect=False)

    assert readings == [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=37.3,
            status="ok",
        )
    ]
    assert [format_temperature_reading(reading) for reading in readings] == [
        "CAM1=37.3 C"
    ]


def test_collect_temperature_readings_reconnects_missing_camera(tmp_path):
    manager = BaslerCameraManager(
        (camera_config(),),
        tmp_path,
        queue.Queue(),
    )
    opened_camera = FakeCamera(41.04)
    opened_names = []

    def open_camera(config):
        opened_names.append(config.name)
        return opened_camera

    manager._open_camera = open_camera

    assert manager.collect_temperature_readings(should_reconnect=True) == [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=41.0,
            status="ok",
        )
    ]
    assert opened_names == ["CAM1"]
    assert manager.cameras["CAM1"] is opened_camera


def test_collect_temperature_readings_marks_missing_without_reconnect(tmp_path):
    manager = BaslerCameraManager(
        (camera_config(),),
        tmp_path,
        queue.Queue(),
    )

    assert manager.collect_temperature_readings(should_reconnect=False) == [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=None,
            status="not detected",
        )
    ]


def test_collect_temperature_readings_reports_unavailable_value(tmp_path):
    manager = BaslerCameraManager(
        (camera_config(),),
        tmp_path,
        queue.Queue(),
    )
    manager.cameras["CAM1"] = BrokenTemperatureCamera()

    readings = manager.collect_temperature_readings(should_reconnect=False)

    assert readings == [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=None,
            status="temperature unavailable",
            error="sensor missing",
        )
    ]
    assert [format_temperature_reading(reading) for reading in readings] == [
        "CAM1=temperature unavailable (sensor missing)"
    ]


def test_runtime_records_temperature_readings_to_file_and_info_log(tmp_path, caplog):
    runtime = CaptureRuntime(runtime_config(tmp_path))
    readings = [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=37.3,
            status="ok",
        ),
        CameraTemperatureReading(
            camera_name="CAM2",
            temperature_c=None,
            status="not detected",
        ),
    ]

    with caplog.at_level("INFO"):
        runtime._record_temperature_readings(readings)

    log_text = (tmp_path / "camera_temp.log").read_text(encoding="utf-8")
    queued_temperature = runtime.temperature_upload_queue.get_nowait()

    assert " | CAM1=37.3 C, CAM2=not detected\n" in log_text
    assert "Camera temperature: CAM1=37.3 C, CAM2=not detected" in caplog.text
    assert queued_temperature.payload["readings"] == [
        {
            "camera_name": "CAM1",
            "temperature_c": 37.3,
            "status": "ok",
        },
        {
            "camera_name": "CAM2",
            "temperature_c": None,
            "status": "not detected",
        },
    ]


def test_runtime_logs_and_uploads_temperature_only_when_readings_change(
    tmp_path,
    caplog,
):
    runtime = CaptureRuntime(runtime_config(tmp_path))
    readings = [
        CameraTemperatureReading(
            camera_name="CAM1",
            temperature_c=37.3,
            status="ok",
        ),
        CameraTemperatureReading(
            camera_name="CAM2",
            temperature_c=41.0,
            status="ok",
        ),
    ]

    with caplog.at_level("INFO"):
        runtime._record_temperature_readings(readings)
        first_upload = runtime.temperature_upload_queue.get_nowait()

        runtime._record_temperature_readings(readings)

        changed_readings = [
            CameraTemperatureReading(
                camera_name="CAM1",
                temperature_c=37.4,
                status="ok",
            ),
            CameraTemperatureReading(
                camera_name="CAM2",
                temperature_c=41.0,
                status="ok",
            ),
        ]
        runtime._record_temperature_readings(changed_readings)
        second_upload = runtime.temperature_upload_queue.get_nowait()

    assert runtime.temperature_upload_queue.empty()
    assert first_upload.payload["readings"][0]["temperature_c"] == 37.3
    assert second_upload.payload["readings"][0]["temperature_c"] == 37.4
    assert caplog.text.count("Camera temperature:") == 2
    assert (
        len(
            (tmp_path / "camera_temp.log")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )


def test_runtime_logs_when_scheduled_time_is_reached(tmp_path, caplog):
    runtime = CaptureRuntime(runtime_config(tmp_path))

    with caplog.at_level("INFO"):
        assert runtime._wait_until_capture(
            start_time=time.monotonic() - 1,
            capture_at=0.5,
            label="CAM1 CAP1",
        )

    assert "scheduled time reached: CAM1 CAP1 at +0.5s" in caplog.text


def runtime_config(tmp_path):
    return RuntimeConfig(
        gpio=GPIOConfig(
            pin=16,
            pull_up=False,
            bounce_time_seconds=0.1,
            high_confirm_seconds=2,
            low_confirm_seconds=5,
        ),
        paths=PathsConfig(
            save_dir=tmp_path,
            camera_temperature_log_file=tmp_path / "camera_temp.log",
        ),
        upload=UploadConfig(
            url="http://receiver.example/upload",
            timeout_seconds=15,
            retry_delay_seconds=2,
        ),
        temperature_upload=UploadConfig(
            url="http://receiver.example/temperature",
            timeout_seconds=15,
            retry_delay_seconds=2,
        ),
        camera_runtime=CameraRuntimeConfig(
            reconnect_interval_seconds=5,
            temperature_log_interval_seconds=10,
        ),
        logging=LoggingConfig(
            level="INFO",
            console=True,
            file=None,
            remove_spam_logs=True,
        ),
        cameras=(camera_config(), camera_config("CAM2", device_index=1)),
    )
