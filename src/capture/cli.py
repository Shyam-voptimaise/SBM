import argparse
import logging
import signal
from typing import Optional, Sequence

from capture.config import ConfigError, load_config
from capture.logging_config import setup_logging
from capture.runtime import CaptureRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SBM camera capture sender.")
    parser.add_argument(
        "--config",
        help=(
            "Path to runtime.yaml. Defaults to SBM_RUNTIME_CONFIG or "
            "./config/runtime.yaml."
        ),
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        setup_logging(config.logging)
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.getLogger(__name__).error("configuration error: %s", exc)
        return 2
    except ValueError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.getLogger(__name__).error("configuration error: %s", exc)
        return 2

    logger = logging.getLogger(__name__)
    app = CaptureRuntime(config)

    def handle_sigterm(signum, _frame) -> None:
        logger.info("shutdown requested by signal %s", signum)
        app.request_stop()

    try:
        signal.signal(signal.SIGTERM, handle_sigterm)
    except (AttributeError, ValueError):
        pass

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("shutdown requested by keyboard interrupt")
        app.request_stop()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(run(argv))
