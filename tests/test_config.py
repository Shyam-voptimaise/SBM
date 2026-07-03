from pathlib import Path

import pytest

from capture.config import CONFIG_ENV_VAR, ConfigError, load_config, resolve_config_path


RUNTIME_YAML = """
gpio:
  pin: 16
  pull_up: false
  bounce_time_seconds: 0.1
  high_confirm_seconds: 2
  low_confirm_seconds: 5
paths:
  save_dir: "~/coil_images"
  camera_temperature_log_file: "~/coil_images/camera_temp.log"
upload:
  url: "http://192.168.0.106:5000/upload"
  timeout_seconds: 15
  retry_delay_seconds: 2
temperature_upload:
  url: "http://192.168.0.106:5000/temperature"
  timeout_seconds: 10
  retry_delay_seconds: 3
camera_runtime:
  reconnect_interval_seconds: 5
  temperature_log_interval_seconds: 10
logging:
  level: "INFO"
  console: true
  file: "~/coil_images/capture.log"
  remove_spam_logs: true
cameras:
  - name: "CAM1"
    device_index: 0
    serial_number: null
    profiles:
      - name: "day"
        exposure_time: 500000.0
        gain_value: 10.0
        start: "06:00"
      - name: "night"
        exposure_time: 500000.0
        gain_value: 10.0
        start: "18:00"
    captures:
      - name: "CAP1"
        delay_after_previous_seconds: 10
      - name: "CAP2"
        delay_after_previous_seconds: 6
  - name: "CAM2"
    device_index: 1
    serial_number: null
    profiles:
      - name: "day"
        exposure_time: 500000.0
        gain_value: 10.0
        start: "06:00"
      - name: "night"
        exposure_time: 500000.0
        gain_value: 10.0
        start: "18:00"
    captures:
      - name: "CAP1"
        delay_after_previous_seconds: 10
      - name: "CAP2"
        delay_after_previous_seconds: 6
"""


def test_load_config_expands_paths_and_loads_production_defaults(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(RUNTIME_YAML, encoding="utf-8")

    config = load_config(str(config_path))

    assert config.gpio.pin == 16
    assert config.gpio.pull_up is False
    assert config.upload.url == "http://192.168.0.106:5000/upload"
    assert config.temperature_upload.url == "http://192.168.0.106:5000/temperature"
    assert config.temperature_upload.timeout_seconds == 10
    assert config.temperature_upload.retry_delay_seconds == 3
    assert config.paths.save_dir == Path("~/coil_images").expanduser()
    assert config.paths.camera_temperature_log_file == Path(
        "~/coil_images/camera_temp.log"
    ).expanduser()
    assert [camera.name for camera in config.cameras] == ["CAM1", "CAM2"]
    assert config.cameras[0].captures[0].delay_after_previous_seconds == 10
    assert config.camera_runtime.temperature_log_interval_seconds == 10
    assert config.camera_runtime.profile_check_interval_seconds == 5.0
    assert [profile.name for profile in config.cameras[0].profiles] == [
        "day",
        "night",
    ]


def test_load_config_parses_day_night_camera_profiles(tmp_path):
    runtime_yaml = RUNTIME_YAML.replace(
        "temperature_log_interval_seconds: 10",
        (
            "temperature_log_interval_seconds: 10\n"
            "  profile_check_interval_seconds: 3"
        ),
    ).replace(
        "      - name: \"night\"\n"
        "        exposure_time: 500000.0\n"
        "        gain_value: 10.0\n"
        "        start: \"18:00\"",
        "      - name: \"night\"\n"
        "        exposure_time: 700000.0\n"
        "        gain_value: 12.0\n"
        "        start: 18:00",
        1,
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(runtime_yaml, encoding="utf-8")

    config = load_config(str(config_path))
    profiles = config.cameras[0].profiles

    assert config.camera_runtime.profile_check_interval_seconds == 3
    assert [(profile.name, profile.start_minutes) for profile in profiles] == [
        ("day", 6 * 60),
        ("night", 18 * 60),
    ]
    assert profiles[1].exposure_time == 700000.0
    assert profiles[1].gain_value == 12.0


def test_load_config_derives_temperature_upload_from_image_upload(tmp_path):
    runtime_yaml = RUNTIME_YAML.replace(
        (
            "temperature_upload:\n"
            "  url: \"http://192.168.0.106:5000/temperature\"\n"
            "  timeout_seconds: 10\n"
            "  retry_delay_seconds: 3\n"
        ),
        "",
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(runtime_yaml, encoding="utf-8")

    config = load_config(str(config_path))

    assert config.temperature_upload.url == "http://192.168.0.106:5000/temperature"
    assert config.temperature_upload.timeout_seconds == config.upload.timeout_seconds
    assert (
        config.temperature_upload.retry_delay_seconds
        == config.upload.retry_delay_seconds
    )


def test_load_config_requires_camera_profiles(tmp_path):
    runtime_yaml = RUNTIME_YAML.replace(
        (
            "    profiles:\n"
            "      - name: \"day\"\n"
            "        exposure_time: 500000.0\n"
            "        gain_value: 10.0\n"
            "        start: \"06:00\"\n"
            "      - name: \"night\"\n"
            "        exposure_time: 500000.0\n"
            "        gain_value: 10.0\n"
            "        start: \"18:00\"\n"
        ),
        "",
        1,
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(runtime_yaml, encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match=r"missing required field: cameras\[0\].profiles",
    ):
        load_config(str(config_path))


def test_environment_config_override(monkeypatch, tmp_path):
    config_path = tmp_path / "custom-runtime.yaml"
    config_path.write_text(RUNTIME_YAML, encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))

    assert resolve_config_path() == config_path
    assert load_config().gpio.pin == 16


def test_default_config_path_uses_config_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resolve_config_path() == tmp_path / "config" / "runtime.yaml"


def test_invalid_config_fails_fast_with_clear_message(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("gpio: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required field: gpio.pin"):
        load_config(str(config_path))
