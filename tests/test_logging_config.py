import logging
from datetime import datetime

import pytest

from capture.logging_config import (
    LOG_FORMAT,
    ProductionLogFormatter,
    setup_fallback_logging,
    setup_logging,
)
from capture.models import LoggingConfig
from capture.time_utils import IST


@pytest.fixture(autouse=True)
def restore_root_logging():
    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level

    yield

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    for handler in previous_handlers:
        root_logger.addHandler(handler)

    root_logger.setLevel(previous_level)


def test_production_log_formatter_includes_operational_context():
    record = logging.LogRecord(
        "capture.runtime",
        logging.INFO,
        "runtime.py",
        131,
        "Camera temperature: %s",
        ("CAM1=37.3 C",),
        None,
    )
    record.created = datetime(2026, 1, 2, 3, 4, 5, 123000, tzinfo=IST).timestamp()
    record.process = 1234
    record.threadName = "sbm-camera-temperature"

    formatted = ProductionLogFormatter(LOG_FORMAT).format(record)

    assert formatted == (
        "2026-01-02T03:04:05.123+05:30 | INFO | "
        "capture.runtime | pid=1234 | "
        "sbm-camera-temperature | runtime.py:131 | "
        "Camera temperature: CAM1=37.3 C"
    )


def test_setup_logging_writes_production_format_to_file(tmp_path):
    log_file = tmp_path / "capture.log"
    setup_logging(
        LoggingConfig(
            level="INFO",
            console=False,
            file=log_file,
            remove_spam_logs=True,
        )
    )

    logging.getLogger("capture.runtime").info("system ready")

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_text = log_file.read_text(encoding="utf-8")

    assert " | INFO | capture.runtime | " in log_text
    assert " | pid=" in log_text
    assert " | MainThread | " in log_text
    assert " | test_logging_config.py:" in log_text
    assert log_text.rstrip().endswith(" | system ready")


def test_setup_fallback_logging_uses_production_format(capsys):
    setup_fallback_logging()

    logging.getLogger("capture.cli").error("configuration error: missing config")

    captured = capsys.readouterr()

    assert " | ERROR | capture.cli | " in captured.err
    assert " | MainThread | " in captured.err
    assert captured.err.rstrip().endswith(" | configuration error: missing config")
