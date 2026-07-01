import os
from datetime import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from capture.models import (
    CameraConfig,
    CameraProfile,
    CameraRuntimeConfig,
    CaptureConfig,
    GPIOConfig,
    LoggingConfig,
    PathsConfig,
    RuntimeConfig,
    UploadConfig,
)

CONFIG_ENV_VAR = "SBM_RUNTIME_CONFIG"
DEFAULT_CONFIG_FILE = Path("config") / "runtime.yaml"


class ConfigError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()

    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    return Path.cwd() / DEFAULT_CONFIG_FILE


def load_config(config_path: Optional[str] = None) -> RuntimeConfig:
    path = resolve_config_path(config_path)

    if not path.exists():
        raise ConfigError(f"runtime config not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to read config/runtime.yaml") from exc

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigError(f"unable to read runtime config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in runtime config {path}: {exc}") from exc

    if raw_config is None:
        raise ConfigError(f"runtime config is empty: {path}")

    data = _require_mapping(raw_config, "runtime config")

    gpio = _load_gpio(_section(data, "gpio"))
    paths = _load_paths(_section(data, "paths"))
    upload = _load_upload(_section(data, "upload"))
    temperature_upload = _load_temperature_upload(
        _optional_section(data, "temperature_upload"),
        upload,
    )
    camera_runtime = _load_camera_runtime(_section(data, "camera_runtime"))
    logging_config = _load_logging(_section(data, "logging"))
    cameras = _load_cameras(_require_sequence(data.get("cameras"), "cameras"))

    return RuntimeConfig(
        gpio=gpio,
        paths=paths,
        upload=upload,
        temperature_upload=temperature_upload,
        camera_runtime=camera_runtime,
        logging=logging_config,
        cameras=tuple(cameras),
    )


def _load_gpio(data: Mapping[str, Any]) -> GPIOConfig:
    return GPIOConfig(
        pin=_require_int(data, "gpio.pin"),
        pull_up=_require_bool(data, "gpio.pull_up"),
        bounce_time_seconds=_require_non_negative_number(
            data,
            "gpio.bounce_time_seconds",
        ),
        high_confirm_seconds=_require_positive_number(
            data,
            "gpio.high_confirm_seconds",
        ),
        low_confirm_seconds=_require_positive_number(
            data,
            "gpio.low_confirm_seconds",
        ),
    )


def _load_paths(data: Mapping[str, Any]) -> PathsConfig:
    return PathsConfig(
        save_dir=_require_path(data, "paths.save_dir"),
        camera_temperature_log_file=_optional_path(
            data,
            "paths.camera_temperature_log_file",
        ),
    )


def _load_upload(data: Mapping[str, Any], section_name: str = "upload") -> UploadConfig:
    return UploadConfig(
        url=_require_non_empty_string(data, f"{section_name}.url"),
        timeout_seconds=_require_positive_number(
            data,
            f"{section_name}.timeout_seconds",
        ),
        retry_delay_seconds=_require_non_negative_number(
            data,
            f"{section_name}.retry_delay_seconds",
        ),
    )


def _load_temperature_upload(
    data: Optional[Mapping[str, Any]],
    upload: UploadConfig,
) -> UploadConfig:
    if data is None:
        return UploadConfig(
            url=_default_temperature_upload_url(upload.url),
            timeout_seconds=upload.timeout_seconds,
            retry_delay_seconds=upload.retry_delay_seconds,
        )

    return _load_upload(data, "temperature_upload")


def _load_camera_runtime(data: Mapping[str, Any]) -> CameraRuntimeConfig:
    return CameraRuntimeConfig(
        reconnect_interval_seconds=_require_positive_number(
            data,
            "camera_runtime.reconnect_interval_seconds",
        ),
        temperature_log_interval_seconds=_require_positive_number(
            data,
            "camera_runtime.temperature_log_interval_seconds",
        ),
        profile_check_interval_seconds=_optional_positive_number(
            data,
            "camera_runtime.profile_check_interval_seconds",
            default=5.0,
        ),
    )


def _load_logging(data: Mapping[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        level=_require_non_empty_string(data, "logging.level").upper(),
        console=_require_bool(data, "logging.console"),
        file=_optional_path(data, "logging.file"),
        remove_spam_logs=_require_bool(data, "logging.remove_spam_logs"),
    )


def _load_cameras(cameras: Sequence[Any]) -> Sequence[CameraConfig]:
    if not cameras:
        raise ConfigError("cameras must contain at least one camera")

    loaded = []
    for index, raw_camera in enumerate(cameras):
        label = f"cameras[{index}]"
        camera = _require_mapping(raw_camera, label)
        captures = _load_captures(
            _require_sequence(camera.get("captures"), f"{label}.captures"),
            label,
        )

        serial_number = camera.get("serial_number")
        if serial_number is not None and not isinstance(serial_number, str):
            raise ConfigError(f"{label}.serial_number must be a string or null")

        profiles = _load_profiles(camera.get("profiles"), label)
        exposure_time = _optional_positive_number(camera, f"{label}.exposure_time")
        gain_value = _optional_non_negative_number(camera, f"{label}.gain_value")

        if profiles:
            exposure_time = (
                exposure_time
                if exposure_time is not None
                else profiles[0].exposure_time
            )
            gain_value = gain_value if gain_value is not None else profiles[0].gain_value
        else:
            exposure_time = _require_positive_number(camera, f"{label}.exposure_time")
            gain_value = _require_non_negative_number(camera, f"{label}.gain_value")
            profiles = (
                CameraProfile(
                    name="default",
                    exposure_time=exposure_time,
                    gain_value=gain_value,
                    start_minutes=0,
                ),
            )

        loaded.append(
            CameraConfig(
                name=_require_non_empty_string(camera, f"{label}.name"),
                device_index=_require_int(camera, f"{label}.device_index"),
                serial_number=serial_number,
                exposure_time=exposure_time,
                gain_value=gain_value,
                captures=tuple(captures),
                profiles=tuple(profiles),
            )
        )

    return loaded


def _load_profiles(raw_profiles: Any, camera_label: str) -> Sequence[CameraProfile]:
    if raw_profiles is None:
        return ()

    profiles = _require_sequence(raw_profiles, f"{camera_label}.profiles")
    if not profiles:
        raise ConfigError(f"{camera_label}.profiles must contain at least one profile")

    loaded = []
    seen_names = set()

    for index, raw_profile in enumerate(profiles):
        label = f"{camera_label}.profiles[{index}]"
        profile = _require_mapping(raw_profile, label)
        name = _require_non_empty_string(profile, f"{label}.name")
        normalized_name = name.lower()

        if normalized_name in seen_names:
            raise ConfigError(f"{label}.name must be unique per camera")
        seen_names.add(normalized_name)

        loaded.append(
            CameraProfile(
                name=name,
                exposure_time=_require_positive_number(
                    profile,
                    f"{label}.exposure_time",
                ),
                gain_value=_require_non_negative_number(
                    profile,
                    f"{label}.gain_value",
                ),
                start_minutes=_parse_start_minutes(
                    _lookup(profile, f"{label}.start"),
                    f"{label}.start",
                ),
            )
        )

    return loaded


def _load_captures(captures: Sequence[Any], camera_label: str) -> Sequence[CaptureConfig]:
    if not captures:
        raise ConfigError(f"{camera_label}.captures must contain at least one capture")

    loaded = []
    for index, raw_capture in enumerate(captures):
        label = f"{camera_label}.captures[{index}]"
        capture = _require_mapping(raw_capture, label)
        delay_key = "delay_after_previous_seconds"
        if delay_key not in capture and "delay_after_previous" in capture:
            delay_key = "delay_after_previous"

        loaded.append(
            CaptureConfig(
                name=_require_non_empty_string(capture, f"{label}.name"),
                delay_after_previous_seconds=_require_non_negative_number(
                    capture,
                    f"{label}.{delay_key}",
                    key=delay_key,
                ),
            )
        )

    return loaded


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in data:
        raise ConfigError(f"missing required section: {name}")
    return _require_mapping(data[name], name)


def _optional_section(data: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    if name not in data or data[name] is None:
        return None
    return _require_mapping(data[name], name)


def _default_temperature_upload_url(upload_url: str) -> str:
    normalized_url = upload_url.rstrip("/")
    if normalized_url.endswith("/upload"):
        return f"{normalized_url[:-len('/upload')]}/temperature"
    return f"{normalized_url}/temperature"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    return value


def _lookup(data: Mapping[str, Any], label: str, key: Optional[str] = None) -> Any:
    field_name = key or label.rsplit(".", 1)[-1]
    if field_name not in data:
        raise ConfigError(f"missing required field: {label}")
    return data[field_name]


def _require_non_empty_string(data: Mapping[str, Any], label: str) -> str:
    value = _lookup(data, label)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _require_int(data: Mapping[str, Any], label: str) -> int:
    value = _lookup(data, label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _require_bool(data: Mapping[str, Any], label: str) -> bool:
    value = _lookup(data, label)
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false")
    return value


def _require_positive_number(
    data: Mapping[str, Any],
    label: str,
    key: Optional[str] = None,
) -> float:
    value = _require_number(data, label, key)
    if value <= 0:
        raise ConfigError(f"{label} must be greater than 0")
    return value


def _optional_positive_number(
    data: Mapping[str, Any],
    label: str,
    default: Optional[float] = None,
    key: Optional[str] = None,
) -> Optional[float]:
    value = _optional_number(data, label, key)
    if value is None:
        return default
    if value <= 0:
        raise ConfigError(f"{label} must be greater than 0")
    return value


def _require_non_negative_number(
    data: Mapping[str, Any],
    label: str,
    key: Optional[str] = None,
) -> float:
    value = _require_number(data, label, key)
    if value < 0:
        raise ConfigError(f"{label} must be greater than or equal to 0")
    return value


def _optional_non_negative_number(
    data: Mapping[str, Any],
    label: str,
    default: Optional[float] = None,
    key: Optional[str] = None,
) -> Optional[float]:
    value = _optional_number(data, label, key)
    if value is None:
        return default
    if value < 0:
        raise ConfigError(f"{label} must be greater than or equal to 0")
    return value


def _require_number(
    data: Mapping[str, Any],
    label: str,
    key: Optional[str] = None,
) -> float:
    value = _lookup(data, label, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return value


def _optional_number(
    data: Mapping[str, Any],
    label: str,
    key: Optional[str] = None,
) -> Optional[float]:
    field_name = key or label.rsplit(".", 1)[-1]
    if field_name not in data:
        return None
    value = data[field_name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return value


def _parse_start_minutes(value: Any, label: str) -> int:
    if isinstance(value, time):
        hour = value.hour
        minute = value.minute
    elif isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value >= 24 * 60:
            raise ConfigError(f"{label} must be a valid 24-hour HH:MM time")
        return value
    elif isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) != 2:
            raise ConfigError(f"{label} must be in HH:MM format")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:
            raise ConfigError(f"{label} must be in HH:MM format") from exc
    else:
        raise ConfigError(f"{label} must be in HH:MM format")

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ConfigError(f"{label} must be a valid 24-hour HH:MM time")

    return hour * 60 + minute


def _require_path(data: Mapping[str, Any], label: str) -> Path:
    value = _lookup(data, label)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string")
    return Path(value).expanduser()


def _optional_path(data: Mapping[str, Any], label: str) -> Optional[Path]:
    value = _lookup(data, label)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a path string or null")
    return Path(value).expanduser()
