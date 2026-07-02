import json
import importlib.util
from io import BytesIO
from pathlib import Path


def load_receiver_module():
    receiver_path = Path(__file__).resolve().parents[1] / "receiver.py"
    spec = importlib.util.spec_from_file_location("receiver_under_test", receiver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receiver_upload_uses_coil_folder_from_metadata(tmp_path, monkeypatch):
    receiver = load_receiver_module()
    monkeypatch.setattr(receiver, "BASE_DIR", str(tmp_path))
    metadata = {
        "coil_no": "145",
        "coil_folder": "COIL_20260701_130434_145",
    }

    response = receiver.app.test_client().post(
        "/upload",
        data={
            "metadata": json.dumps(metadata),
            "image": (
                BytesIO(b"BMfake"),
                "cam_01_cap_01_coil_145.bmp",
            ),
        },
        content_type="multipart/form-data",
    )

    date_folders = list(tmp_path.iterdir())
    saved_folder = date_folders[0] / "COIL_20260701_130434_145"

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert len(date_folders) == 1
    assert (saved_folder / "cam_01_cap_01_coil_145.bmp").exists()
    assert json.loads(
        (saved_folder / "cam_01_cap_01_coil_145.json").read_text(encoding="utf-8")
    ) == metadata


def test_receiver_temperature_endpoint_stores_jsonl(tmp_path, monkeypatch):
    receiver = load_receiver_module()
    monkeypatch.setattr(receiver, "TEMPERATURE_DIR", str(tmp_path))
    payload = {
        "captured_at": "2026-07-01T14:21:30.147+05:30",
        "uploaded_at": "2026-07-01T14:21:31.000+05:30",
        "readings": [
            {
                "camera_name": "CAM1",
                "temperature_c": 58.0,
                "status": "ok",
            },
            {
                "camera_name": "CAM2",
                "temperature_c": 65.0,
                "status": "ok",
            },
        ],
    }

    response = receiver.app.test_client().post("/temperature", json=payload)

    temperature_file = tmp_path / "2026-07-01" / "camera_temperature.jsonl"

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert json.loads(temperature_file.read_text(encoding="utf-8")) == payload


def test_receiver_temperature_endpoint_rejects_missing_readings():
    receiver = load_receiver_module()
    response = receiver.app.test_client().post(
        "/temperature",
        json={"captured_at": "2026-07-01T14:21:30.147+05:30"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing readings list"}
