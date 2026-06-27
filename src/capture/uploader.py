import json
import logging
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from capture.models import QueuedUpload, UploadConfig

LOGGER = logging.getLogger(__name__)


def upload_file(
    file_path: Path,
    metadata: Dict[str, Any],
    config: UploadConfig,
    post: Optional[Callable[..., Any]] = None,
    now: Callable[[], datetime] = datetime.now,
    logger: Optional[logging.Logger] = None,
) -> bool:
    logger = logger or LOGGER

    if post is None:
        import requests

        post = requests.post

    upload_metadata = dict(metadata)
    upload_metadata["uploaded_at"] = now().isoformat()

    try:
        with file_path.open("rb") as image_file:
            files = {
                "image": (
                    file_path.name,
                    image_file,
                    "image/bmp",
                )
            }
            data = {"metadata": json.dumps(upload_metadata)}

            response = post(
                config.url,
                files=files,
                data=data,
                timeout=config.timeout_seconds,
            )

        if response.status_code == 200:
            logger.info("upload succeeded: %s", file_path.name)
            return True

        logger.warning(
            "upload failed with HTTP %s: %s",
            response.status_code,
            file_path.name,
        )
    except Exception:
        logger.exception("upload failed: %s", file_path.name)

    return False


def process_upload_item(
    item: QueuedUpload,
    upload_queue: "queue.Queue[QueuedUpload]",
    config: UploadConfig,
    logger: Optional[logging.Logger] = None,
    post: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
    retry: bool = True,
) -> bool:
    logger = logger or LOGGER

    if upload_file(item.file_path, item.metadata, config, post=post, logger=logger):
        try:
            item.file_path.unlink()
            logger.debug("deleted local file after upload: %s", item.file_path.name)
        except FileNotFoundError:
            logger.warning("uploaded file already missing locally: %s", item.file_path)
        except Exception:
            logger.exception("delete failed after upload: %s", item.file_path)
        return True

    if retry:
        upload_queue.put(item)
        logger.warning("upload failed; requeued for retry: %s", item.file_path.name)
        sleep(config.retry_delay_seconds)

    return False


def uploader_worker(
    upload_queue: "queue.Queue[QueuedUpload]",
    config: UploadConfig,
    stop_event,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or LOGGER

    while not stop_event.is_set():
        try:
            item = upload_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            process_upload_item(item, upload_queue, config, logger=logger)
        finally:
            upload_queue.task_done()

    logger.debug("uploader worker stopped")
