import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from capture.time_utils import ist_date, now_ist

LOGGER = logging.getLogger(__name__)
STATE_FILE_NAME = ".coil_sequence_state.json"
VERBOSE_FILENAME_RE = re.compile(
    r"^cam_\d+_cap_\d+_coil_(\d+)(?:_\d+)?\.bmp$",
    re.IGNORECASE,
)
NUMERIC_FILENAME_RE = re.compile(r"^\d+_\d+_(\d+)(?:_\d+)?\.bmp$", re.IGNORECASE)
LEGACY_FILENAME_RE = re.compile(r"^COIL_(\d+)_.*\.bmp$", re.IGNORECASE)
FOLDER_RE = re.compile(r"^COIL_\d{8}_\d{6}_(?:COIL_)?(\d+)$", re.IGNORECASE)


class CoilSequenceStore:
    def __init__(
        self,
        save_dir: Path,
        state_file: Optional[Path] = None,
        now: Callable[[], datetime] = now_ist,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.state_file = state_file or self.save_dir / STATE_FILE_NAME
        self._now = now
        self.logger = logger or LOGGER
        self._lock = threading.Lock()
        self._state_date = ""
        self._last_coil_number = 0
        self._load_or_recover()

    def next_coil_number(self) -> str:
        with self._lock:
            self._last_coil_number += 1
            self._write_state()
            return format_coil_number(self._last_coil_number)

    def refresh_current_date(self) -> None:
        with self._lock:
            current_date = self._current_date()
            if current_date == self._state_date:
                return

            self._state_date = current_date
            self._last_coil_number = self._scan_latest_coil_number(current_date)
            self._write_state()
            self.logger.info(
                "coil sequence date refreshed for %s; latest saved coil=%s",
                current_date,
                format_coil_number(self._last_coil_number),
            )

    def _load_or_recover(self) -> None:
        current_date = self._current_date()
        state = self._read_state()
        scanned_last = self._scan_latest_coil_number(current_date)

        if state and state.get("ist_date") == current_date:
            state_last = _coerce_non_negative_int(state.get("last_coil_number"))
            self._last_coil_number = max(state_last, scanned_last)
        else:
            self._last_coil_number = scanned_last

        self._state_date = current_date
        self._write_state()

    def _current_date(self) -> str:
        return ist_date(self._now())

    def _read_state(self) -> Optional[dict[str, Any]]:
        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                state = json.load(file)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "could not read coil sequence state; recovering from saved files",
                exc_info=True,
            )
            return None

        if not isinstance(state, dict):
            self.logger.warning("invalid coil sequence state; recovering from files")
            return None

        return state

    def _write_state(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_file.with_name(f"{self.state_file.name}.tmp")
        state = {
            "ist_date": self._state_date,
            "last_coil_number": self._last_coil_number,
            "updated_at": self._now().isoformat(),
        }

        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)
            file.write("\n")

        os.replace(tmp_path, self.state_file)

    def _scan_latest_coil_number(self, date_folder: str) -> int:
        folder = self.save_dir / date_folder
        if not folder.exists():
            return 0

        latest = 0

        try:
            children = list(folder.iterdir())
        except OSError:
            self.logger.warning("could not scan coil folder: %s", folder, exc_info=True)
            return 0

        for child in children:
            if child.is_dir():
                latest = max(latest, _extract_folder_coil_number(child.name))

        try:
            files = folder.rglob("*.bmp")
            for file_path in files:
                latest = max(latest, _extract_filename_coil_number(file_path.name))
        except OSError:
            self.logger.warning(
                "could not scan saved BMP files: %s",
                folder,
                exc_info=True,
            )

        return latest


def format_coil_number(value: int) -> str:
    if value < 0:
        raise ValueError("coil number cannot be negative")
    return f"{value:02d}"


def _extract_filename_coil_number(filename: str) -> int:
    for pattern in (VERBOSE_FILENAME_RE, NUMERIC_FILENAME_RE, LEGACY_FILENAME_RE):
        match = pattern.match(filename)
        if match:
            return _coerce_non_negative_int(match.group(1))
    return 0


def _extract_folder_coil_number(folder_name: str) -> int:
    match = FOLDER_RE.match(folder_name)
    if not match:
        return 0
    return _coerce_non_negative_int(match.group(1))


def _coerce_non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)
