import logging
import queue
import threading
import time
from typing import Optional

from capture.camera import BaslerCameraManager
from capture.capture import (
    build_capture_schedule,
    build_coil_folder_name,
)
from capture.gpio import create_trigger
from capture.models import QueuedUpload, RuntimeConfig
from capture.sequence import CoilSequenceStore
from capture.time_utils import now_ist
from capture.uploader import uploader_worker

LOGGER = logging.getLogger(__name__)
POLL_INTERVAL_SECONDS = 0.05


class CaptureRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self.upload_queue: "queue.Queue[QueuedUpload]" = queue.Queue()
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

        temperature_thread = threading.Thread(
            target=self._temperature_worker,
            daemon=True,
            name="sbm-camera-temperature",
        )
        temperature_thread.start()
        self._threads.append(temperature_thread)

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

            self._write_temperature_log(readings)
            self.logger.debug("camera temperature: %s", ", ".join(readings))

            self.stop_event.wait(
                self.config.camera_runtime.temperature_log_interval_seconds
            )

    def _write_temperature_log(self, readings) -> None:
        log_file = self.config.paths.camera_temperature_log_file
        if log_file is None:
            return

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = now_ist().isoformat(timespec="seconds")
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
