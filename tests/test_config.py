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
camera_runtime:
  reconnect_interval_seconds: 5
  temperature_log_interval_seconds: 1
logging:
  level: "INFO"
  console: true
  file: "~/coil_images/capture.log"
  remove_spam_logs: true
cameras:
  - name: "CAM1"
    device_index: 0
    serial_number: null
    exposure_time: 500000.0
    gain_value: 10.0
    captures:
      - name: "CAP1"
        delay_after_previous_seconds: 10
      - name: "CAP2"
        delay_after_previous_seconds: 6
  - name: "CAM2"
    device_index: 1
    serial_number: null
    exposure_time: 500000.0
    gain_value: 10.0
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
    assert config.paths.save_dir == Path("~/coil_images").expanduser()
    assert config.paths.camera_temperature_log_file == Path(
        "~/coil_images/camera_temp.log"
    ).expanduser()
    assert [camera.name for camera in config.cameras] == ["CAM1", "CAM2"]
    assert config.cameras[0].captures[0].delay_after_previous_seconds == 10


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
