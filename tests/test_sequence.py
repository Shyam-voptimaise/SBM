import json
from datetime import datetime

from capture.sequence import CoilSequenceStore
from capture.time_utils import IST


def fixed_now(year=2026, month=1, day=2, hour=3):
    return lambda: datetime(year, month, day, hour, 4, 5, tzinfo=IST)


def test_sequence_starts_at_01_for_new_ist_day(tmp_path):
    store = CoilSequenceStore(tmp_path, now=fixed_now())

    assert store.next_coil_number() == "01"
    assert store.next_coil_number() == "02"


def test_sequence_continues_from_state_after_restart_same_ist_day(tmp_path):
    state_file = tmp_path / ".coil_sequence_state.json"
    first_store = CoilSequenceStore(tmp_path, state_file=state_file, now=fixed_now())

    assert first_store.next_coil_number() == "01"
    assert first_store.next_coil_number() == "02"

    restarted_store = CoilSequenceStore(
        tmp_path,
        state_file=state_file,
        now=fixed_now(),
    )

    assert restarted_store.next_coil_number() == "03"


def test_sequence_resets_for_new_ist_day(tmp_path):
    state_file = tmp_path / ".coil_sequence_state.json"
    store = CoilSequenceStore(tmp_path, state_file=state_file, now=fixed_now())
    assert store.next_coil_number() == "01"

    next_day_store = CoilSequenceStore(
        tmp_path,
        state_file=state_file,
        now=fixed_now(day=3),
    )

    assert next_day_store.next_coil_number() == "01"


def test_sequence_recovers_from_latest_saved_file_when_state_missing(tmp_path):
    saved = tmp_path / "2026-01-02" / "COIL_20260102_030405_07"
    saved.mkdir(parents=True)
    (saved / "cam_01_cap_01_coil_07.bmp").write_bytes(b"BMfake")

    store = CoilSequenceStore(tmp_path, now=fixed_now())

    assert store.next_coil_number() == "08"


def test_sequence_uses_max_of_state_and_saved_files(tmp_path):
    date_folder = tmp_path / "2026-01-02"
    saved = date_folder / "COIL_20260102_030405_04"
    saved.mkdir(parents=True)
    (saved / "cam_01_cap_01_coil_04.bmp").write_bytes(b"BMfake")
    (tmp_path / ".coil_sequence_state.json").write_text(
        json.dumps({"ist_date": "2026-01-02", "last_coil_number": 10}),
        encoding="utf-8",
    )

    store = CoilSequenceStore(tmp_path, now=fixed_now())

    assert store.next_coil_number() == "11"


def test_sequence_recovers_from_corrupt_state_file(tmp_path):
    state_file = tmp_path / ".coil_sequence_state.json"
    state_file.write_text("not json", encoding="utf-8")
    saved = tmp_path / "2026-01-02" / "COIL_20260102_030405_03"
    saved.mkdir(parents=True)
    (saved / "cam_01_cap_01_coil_03.bmp").write_bytes(b"BMfake")

    store = CoilSequenceStore(tmp_path, state_file=state_file, now=fixed_now())

    assert store.next_coil_number() == "04"
