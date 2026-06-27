import json
import queue
from datetime import datetime

from capture.models import QueuedUpload, UploadConfig
from capture.uploader import process_upload_item, upload_file


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


def upload_config():
    return UploadConfig(
        url="http://receiver.example/upload",
        timeout_seconds=15,
        retry_delay_seconds=2,
    )


def test_upload_file_uses_receiver_contract_and_adds_uploaded_at(tmp_path):
    image_path = tmp_path / "cam_01_cap_01_coil_01.bmp"
    image_path.write_bytes(b"BMfake")
    calls = []

    def fake_post(url, files, data, timeout):
        image_field = files["image"]
        calls.append(
            {
                "url": url,
                "filename": image_field[0],
                "content": image_field[1].read(),
                "content_type": image_field[2],
                "data": data,
                "timeout": timeout,
            }
        )
        return Response(200)

    ok = upload_file(
        image_path,
        {"coil_no": "01"},
        upload_config(),
        post=fake_post,
        now=lambda: datetime(2026, 1, 2, 3, 4, 5),
    )

    assert ok is True
    assert calls == [
        {
            "url": "http://receiver.example/upload",
            "filename": "cam_01_cap_01_coil_01.bmp",
            "content": b"BMfake",
            "content_type": "image/bmp",
            "data": {
                "metadata": json.dumps(
                    {
                        "coil_no": "01",
                        "uploaded_at": "2026-01-02T03:04:05",
                    }
                )
            },
            "timeout": 15,
        }
    ]


def test_process_upload_item_deletes_file_after_success(tmp_path):
    image_path = tmp_path / "image.bmp"
    image_path.write_bytes(b"BMfake")
    retry_queue = queue.Queue()
    item = QueuedUpload(image_path, {"coil_no": "01"})

    ok = process_upload_item(
        item,
        retry_queue,
        upload_config(),
        post=lambda *_args, **_kwargs: Response(200),
    )

    assert ok is True
    assert not image_path.exists()
    assert retry_queue.empty()


def test_process_upload_item_requeues_and_keeps_file_after_failure(tmp_path):
    image_path = tmp_path / "image.bmp"
    image_path.write_bytes(b"BMfake")
    retry_queue = queue.Queue()
    item = QueuedUpload(image_path, {"coil_no": "01"})

    ok = process_upload_item(
        item,
        retry_queue,
        upload_config(),
        post=lambda *_args, **_kwargs: Response(500),
        sleep=lambda _seconds: None,
    )

    assert ok is False
    assert image_path.exists()
    assert retry_queue.get_nowait() == item
