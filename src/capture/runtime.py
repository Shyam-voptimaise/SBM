import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Sequence, Tuple

from capture.camera import BaslerCameraManager, format_temperature_reading
from capture.capture import (
    build_capture_schedule,
    build_coil_folder_name,
)
from capture.gpio import create_trigger
from capture.models import (
    CameraTemperatureReading,
    QueuedTemperatureUpload,
    QueuedUpload,
    RuntimeConfig,
)
from capture.sequence import CoilSequenceStore
from capture.time_utils import now_ist, seconds_until_next_ist_midnight
from capture.uploader import temperature_uploader_worker, uploader_worker

LOGGER = logging.getLogger(__name__)
POLL_INTERVAL_SECONDS = 0.05
PROFILE_IDLE_WAIT_SECONDS = 24 * 60 * 60
MIN_SCHEDULED_WAIT_SECONDS = 0.1


class CaptureRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self.upload_queue: "queue.Queue[QueuedUpload]" = queue.Queue()
        self.temperature_upload_queue: "queue.Queue[QueuedTemperatureUpload]" = (
            queue.Queue()
        )
        self.stop_event = threading.Event()
        self.sequence_store = CoilSequenceStore(
            self.config.paths.save_dir,
            logger=self.logger,
        )
        self.camera_manager = BaslerCameraManager(
            self.config.cameras,
            self.config.paths.save_dir,
            self.upload_queue,
            logger=self.logger,
        )
        self._threads = []
        self._trigger = None
        self._last_temperature_upload_signature: Optional[
            Tuple[Tuple[str, Optional[float], str], ...]
        ] = None

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.config.paths.save_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.camera_manager.open_configured_cameras()
            self.camera_manager.log_missing_camera_status()
            self._start_worker_threads()

            self._trigger = create_trigger(self.config.gpio)
            self.logger.info("system ready")
            self._run_gpio_loop(self._trigger)
        finally:
            self.request_stop()
            self._close_trigger()
            self.camera_manager.close_all()
            self.logger.info("shutdown")

    def _start_worker_threads(self) -> None:
        upload_thread = threading.Thread(
            target=uploader_worker,
            args=(
                self.upload_queue,
                self.config.upload,
                self.stop_event,
                self.logger,
            ),
            daemon=True,
            name="sbm-uploader",
        )
        upload_thread.start()
        self._threads.append(upload_thread)

        temperature_upload_thread = threading.Thread(
            target=temperature_uploader_worker,
            args=(
                self.temperature_upload_queue,
                self.config.temperature_upload,
                self.stop_event,
                self.logger,
            ),
            daemon=True,
            name="sbm-temperature-uploader",
        )
        temperature_upload_thread.start()
        self._threads.append(temperature_upload_thread)

        temperature_thread = threading.Thread(
            target=self._temperature_worker,
            daemon=True,
            name="sbm-camera-temperature",
        )
        temperature_thread.start()
        self._threads.append(temperature_thread)

        profile_thread = threading.Thread(
            target=self._profile_worker,
            daemon=True,
            name="sbm-camera-profiles",
        )
        profile_thread.start()
        self._threads.append(profile_thread)

        sequence_thread = threading.Thread(
            target=self._coil_sequence_date_worker,
            daemon=True,
            name="sbm-coil-sequence-date",
        )
        sequence_thread.start()
        self._threads.append(sequence_thread)

    def _profile_worker(self) -> None:
        while not self.stop_event.is_set():
            wait_seconds = self.config.camera_runtime.profile_check_interval_seconds

            try:
                self.camera_manager.refresh_active_profiles()
                wait_seconds = (
                    self.camera_manager.seconds_until_next_profile_change()
                )
                if wait_seconds is None:
                    wait_seconds = PROFILE_IDLE_WAIT_SECONDS
            except Exception:
                self.logger.exception("camera profile refresh failed")

            self.stop_event.wait(
                max(wait_seconds, MIN_SCHEDULED_WAIT_SECONDS)
            )

    def _coil_sequence_date_worker(self) -> None:
        while not self.stop_event.is_set():
            wait_seconds = seconds_until_next_ist_midnight(now_ist())

            if self.stop_event.wait(
                max(wait_seconds, MIN_SCHEDULED_WAIT_SECONDS)
            ):
                return

            try:
                self.sequence_store.refresh_current_date()
            except Exception:
                self.logger.exception("coil sequence date refresh failed")
                self.stop_event.wait(
                    self.config.camera_runtime.profile_check_interval_seconds
                )

    def _temperature_worker(self) -> None:
        last_reconnect_attempt = time.monotonic()

        while not self.stop_event.is_set():
            now = time.monotonic()
            should_reconnect = (
                now - last_reconnect_attempt
                >= self.config.camera_runtime.reconnect_interval_seconds
            )

            readings = self.camera_manager.collect_temperature_readings(
                should_reconnect
            )
            if should_reconnect:
                last_reconnect_attempt = now

            self._record_temperature_readings(readings)

            self.stop_event.wait(
                self.config.camera_runtime.temperature_log_interval_seconds
            )

    def _record_temperature_readings(
        self,
        readings: Sequence[CameraTemperatureReading],
    ) -> None:
        captured_at = now_ist()
        formatted_readings = [
            format_temperature_reading(reading) for reading in readings
        ]

        self._write_temperature_log(formatted_readings, captured_at)
        self._queue_temperature_upload_if_changed(readings, captured_at)
        self.logger.info("Camera temperature: %s", ", ".join(formatted_readings))

    def _queue_temperature_upload_if_changed(
        self,
        readings: Sequence[CameraTemperatureReading],
        captured_at: datetime,
    ) -> None:
        signature = _temperature_upload_signature(readings)
        if signature == self._last_temperature_upload_signature:
            self.logger.debug("camera temperature unchanged; upload skipped")
            return

        payload = {
            "captured_at": captured_at.isoformat(timespec="milliseconds"),
            "readings": [
                _temperature_reading_to_payload(reading) for reading in readings
            ],
        }
        self.temperature_upload_queue.put(QueuedTemperatureUpload(payload))
        self._last_temperature_upload_signature = signature

    def _write_temperature_log(
        self,
        readings: Sequence[str],
        captured_at: datetime,
    ) -> None:
        log_file = self.config.paths.camera_temperature_log_file
        if log_file is None:
            return

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = captured_at.isoformat(timespec="seconds")
            line = f"{timestamp} | " + ", ".join(readings)
            with log_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except Exception:
            self.logger.exception("failed to write camera temperature log")

    def _run_gpio_loop(self, trigger) -> None:
        state_idle = "idle"
        state_confirm_high = "confirm_high"
        state_wait_low = "wait_low"

        state = state_idle
        high_start = None
        low_start = None
        last_high_progress_second = None
        last_low_progress_second = None

        while not self.stop_event.is_set():
            current_value = trigger.value
            now = time.monotonic()

            if state == state_idle:
                if current_value:
                    high_start = now
                    state = state_confirm_high
                    last_high_progress_second = None
                    self.logger.info("high detected")
                self.stop_event.wait(POLL_INTERVAL_SECONDS)
                continue

            if state == state_confirm_high:
                if current_value:
                    elapsed = now - high_start
                    elapsed_second = int(elapsed)
                    if elapsed_second != last_high_progress_second:
                        self.logger.debug(
                            "high confirm %.1f/%.1fs",
                            elapsed,
                            self.config.gpio.high_confirm_seconds,
                        )
                        last_high_progress_second = elapsed_second

                    if elapsed >= self.config.gpio.high_confirm_seconds:
                        self.logger.info(
                            "coil confirmed at %s",
                            now_ist().strftime("%H:%M:%S"),
                        )
                        self._process_coil()
                        state = state_wait_low
                        low_start = None
                else:
                    self.logger.debug("high interrupted")
                    state = state_idle
                    high_start = None
                    last_high_progress_second = None

                self.stop_event.wait(POLL_INTERVAL_SECONDS)
                continue

            if state == state_wait_low:
                if not current_value:
                    if low_start is None:
                        low_start = now
                        last_low_progress_second = None
                        self.logger.debug("low detected")

                    elapsed = now - low_start
                    elapsed_second = int(elapsed)
                    if elapsed_second != last_low_progress_second:
                        self.logger.debug(
                            "low confirm %.1f/%.1fs",
                            elapsed,
                            self.config.gpio.low_confirm_seconds,
                        )
                        last_low_progress_second = elapsed_second

                    if elapsed >= self.config.gpio.low_confirm_seconds:
                        self.logger.info("system ready")
                        state = state_idle
                        high_start = None
                        low_start = None
                        last_high_progress_second = None
                        last_low_progress_second = None
                else:
                    low_start = None
                    last_low_progress_second = None

                self.stop_event.wait(POLL_INTERVAL_SECONDS)

    def _process_coil(self) -> None:
        coil_no = self.sequence_store.next_coil_number()
        coil_started_at = now_ist()
        coil_folder = build_coil_folder_name(coil_no, coil_started_at)
        schedule = build_capture_schedule(self.config.cameras)
        start_time = time.monotonic()

        self.logger.info("processing %s -> %s", coil_no, coil_folder)

        for scheduled_capture in schedule:
            camera = scheduled_capture.camera
            capture = scheduled_capture.capture
            label = f"{camera.name} {capture.name}"
            self.logger.debug(
                "%s scheduled at +%ss",
                label,
                scheduled_capture.capture_at_seconds,
            )

            if not self._wait_until_capture(
                start_time,
                scheduled_capture.capture_at_seconds,
                label,
            ):
                return

            self.logger.info("capture started: %s", label)
            self.camera_manager.capture_scheduled(
                coil_no,
                coil_folder,
                coil_started_at,
                camera,
                capture,
            )

    def _wait_until_capture(
        self,
        start_time: float,
        capture_at: float,
        label: str,
    ) -> bool:
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - start_time
            remaining = capture_at - elapsed

            if remaining <= 0:
                return True

            self.logger.debug("%s in %ss", label, int(remaining + 0.999))
            self.stop_event.wait(min(1, remaining))

        return False

    def _close_trigger(self) -> None:
        if self._trigger is None:
            return

        close = getattr(self._trigger, "close", None)
        if close is None:
            return

        try:
            close()
        except Exception:
            self.logger.exception("failed to close GPIO trigger")


def _temperature_reading_to_payload(reading: CameraTemperatureReading) -> dict:
    payload = {
        "camera_name": reading.camera_name,
        "temperature_c": reading.temperature_c,
        "status": reading.status,
    }

    if reading.error:
        payload["error"] = reading.error

    return payload


def _temperature_upload_signature(
    readings: Sequence[CameraTemperatureReading],
) -> Tuple[Tuple[str, Optional[float], str], ...]:
    return tuple(
        (reading.camera_name, reading.temperature_c, reading.status)
        for reading in readings
    )
