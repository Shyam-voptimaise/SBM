import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from capture.capture import build_capture_metadata, build_image_filename
from capture.models import CameraConfig, CaptureConfig, QueuedUpload

LOGGER = logging.getLogger(__name__)


class BaslerCameraManager:
    def __init__(
        self,
        camera_configs: Sequence[CameraConfig],
        save_dir: Path,
        upload_queue: "queue.Queue[QueuedUpload]",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.camera_configs = tuple(camera_configs)
        self.save_dir = save_dir
        self.upload_queue = upload_queue
        self.logger = logger or LOGGER
        self.lock = threading.Lock()
        self.cameras: Dict[str, object] = {
            camera.name: None for camera in self.camera_configs
        }

    def open_configured_cameras(self) -> None:
        with self.lock:
            for camera_config in self.camera_configs:
                if self.cameras.get(camera_config.name) is not None:
                    continue
                self.cameras[camera_config.name] = self._open_camera(camera_config)

    def log_missing_camera_status(self) -> None:
        with self.lock:
            for camera_config in self.camera_configs:
                if self.cameras.get(camera_config.name) is None:
                    self.logger.warning(
                        "%s: camera not detected; system will continue",
                        camera_config.name,
                    )

    def capture_scheduled(
        self,
        coil_no: str,
        coil_folder: str,
        coil_started_at: datetime,
        camera_config: CameraConfig,
        capture_config: CaptureConfig,
    ) -> bool:
        with self.lock:
            camera = self._get_or_open_camera(camera_config)
            if camera is None:
                self.logger.warning(
                    "%s %s: skipped because camera is unavailable",
                    camera_config.name,
                    capture_config.name,
                )
                return False

            success = self._capture_image(
                camera,
                coil_no,
                coil_folder,
                coil_started_at,
                camera_config,
                capture_config,
            )

            if not success:
                self._close_camera(camera, camera_config.name)
                self.cameras[camera_config.name] = self._open_camera(camera_config)

            return success

    def collect_temperature_readings(self, should_reconnect: bool) -> List[str]:
        readings = []

        with self.lock:
            for camera_config in self.camera_configs:
                camera_name = camera_config.name
                camera = self.cameras.get(camera_name)

                if camera is None and should_reconnect:
                    camera = self._open_camera(camera_config)
                    self.cameras[camera_name] = camera

                if camera is None:
                    readings.append(f"{camera_name}=not detected")
                    continue

                try:
                    temperature = camera.TemperatureAbs.Value
                    readings.append(f"{camera_name}={temperature:.1f} C")
                except Exception as exc:
                    readings.append(
                        f"{camera_name}=temperature unavailable ({exc})"
                    )

        return readings

    def close_all(self) -> None:
        with self.lock:
            for camera_name, camera in self.cameras.items():
                self._close_camera(camera, camera_name)

    def _get_or_open_camera(self, camera_config: CameraConfig):
        camera = self.cameras.get(camera_config.name)
        if camera is None:
            camera = self._open_camera(camera_config)
            self.cameras[camera_config.name] = camera
        return camera

    def _open_camera(self, camera_config: CameraConfig):
        pylon = _load_pylon()
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()

        if not devices:
            self.logger.warning("no Basler cameras found")
            return None

        device = self._find_camera_device(devices, camera_config)
        if device is None:
            return None

        camera = pylon.InstantCamera(factory.CreateDevice(device))

        try:
            self._configure_camera(camera, camera_config, pylon)
        except Exception as exc:
            self.logger.error(
                "%s: camera open/config error: %s",
                camera_config.name,
                exc,
            )
            self._close_camera(camera, camera_config.name)
            return None

        self.logger.info(
            "%s: camera connected (%s)",
            camera_config.name,
            self._get_device_label(device),
        )
        return camera

    def _configure_camera(self, camera, camera_config: CameraConfig, pylon) -> None:
        camera.Open()
        camera.AcquisitionMode.SetValue("Continuous")

        camera.ExposureAuto.SetValue("Off")
        try:
            camera.ExposureTime.SetValue(camera_config.exposure_time)
        except Exception:
            camera.ExposureTimeAbs.SetValue(camera_config.exposure_time)

        try:
            camera.GainAuto.SetValue("Off")
            camera.Gain.SetValue(camera_config.gain_value)
        except Exception:
            try:
                camera.GainRaw.SetValue(int(camera_config.gain_value))
            except Exception:
                self.logger.debug(
                    "%s: gain setting not supported",
                    camera_config.name,
                )

        camera.TriggerMode.SetValue("On")
        camera.TriggerSource.SetValue("Software")
        camera.TriggerSelector.SetValue("FrameStart")
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    def _find_camera_device(self, devices, camera_config: CameraConfig):
        if camera_config.serial_number:
            for device in devices:
                try:
                    if device.GetSerialNumber() == camera_config.serial_number:
                        return device
                except Exception:
                    continue

            self.logger.warning(
                "%s: serial number %s not found",
                camera_config.name,
                camera_config.serial_number,
            )
            return None

        if camera_config.device_index < 0 or camera_config.device_index >= len(devices):
            self.logger.warning(
                "%s: device index %s not found; %s Basler camera(s) detected",
                camera_config.name,
                camera_config.device_index,
                len(devices),
            )
            return None

        return devices[camera_config.device_index]

    def _capture_image(
        self,
        camera,
        coil_no: str,
        coil_folder: str,
        coil_started_at: datetime,
        camera_config: CameraConfig,
        capture_config: CaptureConfig,
    ) -> bool:
        pylon = _load_pylon()
        result = None
        image = None

        try:
            camera.ExecuteSoftwareTrigger()
            result = camera.RetrieveResult(5000)

            if not result.GrabSucceeded():
                self.logger.warning(
                    "%s %s: grab failed",
                    camera_config.name,
                    capture_config.name,
                )
                return False

            date_folder = coil_started_at.strftime("%Y-%m-%d")
            folder = self.save_dir / date_folder / coil_folder
            folder.mkdir(parents=True, exist_ok=True)

            filename_timestamp = datetime.now()
            filename = build_image_filename(
                coil_no,
                camera_config.name,
                capture_config.name,
                filename_timestamp,
            )
            path = folder / filename

            image = pylon.PylonImage()
            image.AttachGrabResultBuffer(result)
            image.Save(pylon.ImageFileFormat_Bmp, str(path))

            captured_at = datetime.now()
            metadata = build_capture_metadata(
                coil_no,
                coil_folder,
                coil_started_at,
                camera_config,
                capture_config,
                captured_at,
            )
            self.upload_queue.put(QueuedUpload(path, metadata))

            self.logger.info(
                "capture saved: %s %s -> %s",
                camera_config.name,
                capture_config.name,
                filename,
            )
            return True
        except Exception:
            self.logger.exception(
                "%s %s: camera capture error",
                camera_config.name,
                capture_config.name,
            )
            return False
        finally:
            if image is not None:
                try:
                    image.Release()
                except Exception:
                    pass

            if result is not None:
                try:
                    result.Release()
                except Exception:
                    pass

    def _close_camera(self, camera, camera_name: str) -> None:
        if camera is None:
            return

        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass

        try:
            if camera.IsOpen():
                camera.Close()
                self.logger.info("%s: camera disconnected", camera_name)
        except Exception:
            pass

    def _get_device_label(self, device) -> str:
        details = []

        try:
            details.append(device.GetModelName())
        except Exception:
            pass

        try:
            details.append(f"SN {device.GetSerialNumber()}")
        except Exception:
            pass

        return " / ".join(details) if details else "unknown device"


def _load_pylon():
    from pypylon import pylon

    return pylon
