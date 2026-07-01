import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from capture.models import LoggingConfig
from capture.time_utils import IST

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "pid=%(process)d | %(threadName)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
DEFAULT_MAX_LOG_BYTES = 5_000_000
DEFAULT_BACKUP_COUNT = 3


class ProductionLogFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None) -> str:
        return datetime.fromtimestamp(record.created, IST).isoformat(
            timespec="milliseconds"
        )


def setup_logging(config: LoggingConfig) -> None:
    level = getattr(logging, config.level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"invalid logging level: {config.level}")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = ProductionLogFormatter(LOG_FORMAT)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if config.file is not None:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.file,
            maxBytes=DEFAULT_MAX_LOG_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

    if config.remove_spam_logs:
        for logger_name in ("urllib3", "requests", "werkzeug"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_fallback_logging() -> None:
    logging.basicConfig(
        level=logging.ERROR,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(ProductionLogFormatter(LOG_FORMAT))
