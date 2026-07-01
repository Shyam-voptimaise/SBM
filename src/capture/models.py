from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CaptureConfig:
    name: str
    delay_after_previous_seconds: float


@dataclass(frozen=True)
class CameraProfile:
    name: str
    exposure_time: float
    gain_value: float
    start_minutes: int


@dataclass(frozen=True)
class CameraConfig:
    name: str
    device_index: int
    serial_number: Optional[str]
    exposure_time: float
    gain_value: float
    captures: Tuple[CaptureConfig, ...]
    profiles: Tuple[CameraProfile, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GPIOConfig:
    pin: int
    pull_up: bool
    bounce_time_seconds: float
    high_confirm_seconds: float
    low_confirm_seconds: float


@dataclass(frozen=True)
class PathsConfig:
    save_dir: Path
    camera_temperature_log_file: Optional[Path]


@dataclass(frozen=True)
class UploadConfig:
    url: str
    timeout_seconds: float
    retry_delay_seconds: float


@dataclass(frozen=True)
class CameraRuntimeConfig:
    reconnect_interval_seconds: float
    temperature_log_interval_seconds: float
    profile_check_interval_seconds: float = 5.0


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    console: bool
    file: Optional[Path]
    remove_spam_logs: bool


@dataclass(frozen=True)
class RuntimeConfig:
    gpio: GPIOConfig
    paths: PathsConfig
    upload: UploadConfig
    temperature_upload: UploadConfig
    camera_runtime: CameraRuntimeConfig
    logging: LoggingConfig
    cameras: Tuple[CameraConfig, ...]


@dataclass(frozen=True)
class ScheduledCapture:
    capture_at_seconds: float
    camera: CameraConfig
    capture: CaptureConfig


@dataclass(frozen=True)
class QueuedUpload:
    file_path: Path
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CameraTemperatureReading:
    camera_name: str
    temperature_c: Optional[float]
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class QueuedTemperatureUpload:
    payload: Dict[str, Any]
