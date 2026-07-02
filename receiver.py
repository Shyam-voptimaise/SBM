from flask import Flask, request, jsonify
from waitress import serve

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__)

BASE_DIR = "received_images"
TEMPERATURE_DIR = "received_temperatures"
LOG_DIR = "logs"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TEMPERATURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Industrial Logging Setup
# ---------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(process)d | %(threadName)s | "
    "%(module)s.%(funcName)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("receiver_server")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "receiver_server.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


@app.route("/")
def home():
    logger.info("Health check endpoint accessed")
    return "Receiver Running"


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            logger.warning("Upload rejected: no image file in request")
            return jsonify({"error": "No image"}), 400

        image = request.files["image"]

        if image.filename == "":
            logger.warning("Upload rejected: empty image filename")
            return jsonify({"error": "Empty filename"}), 400

        metadata_str = request.form.get("metadata", "{}")

        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            logger.warning("Upload rejected: invalid metadata JSON")
            return jsonify({"error": "Invalid metadata JSON"}), 400

        coil_folder = str(metadata.get("coil_folder") or "")
        folder_name = secure_filename(coil_folder)

        if not folder_name:
            coil_no = str(metadata.get("coil_no", "UNKNOWN"))
            folder_name = secure_filename(coil_no) or "UNKNOWN"

        date_folder = datetime.now().strftime("%Y-%m-%d")

        save_folder = os.path.join(
            BASE_DIR,
            date_folder,
            folder_name
        )

        os.makedirs(save_folder, exist_ok=True)

        safe_filename = secure_filename(image.filename)

        image_path = os.path.join(
            save_folder,
            safe_filename
        )

        image.save(image_path)

        json_name = (
            os.path.splitext(safe_filename)[0]
            + ".json"
        )

        metadata_path = os.path.join(
            save_folder,
            json_name
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(
            "Image upload completed | folder_name=%s | image_path=%s | metadata_path=%s",
            folder_name,
            image_path,
            metadata_path
        )

        return jsonify({"status": "ok"}), 200

    except Exception:
        logger.exception("Image upload failed due to unexpected error")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/temperature", methods=["POST"])
def temperature():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            logger.warning("Temperature upload rejected: invalid JSON payload")
            return jsonify({"error": "Invalid JSON payload"}), 400

        readings = payload.get("readings")

        if not isinstance(readings, list):
            logger.warning("Temperature upload rejected: missing readings list")
            return jsonify({"error": "Missing readings list"}), 400

        captured_at = payload.get("captured_at")

        if isinstance(captured_at, str) and len(captured_at) >= 10:
            date_folder = captured_at[:10]
        else:
            date_folder = datetime.now().strftime("%Y-%m-%d")

        save_folder = os.path.join(
            TEMPERATURE_DIR,
            date_folder
        )

        os.makedirs(save_folder, exist_ok=True)

        temperature_path = os.path.join(
            save_folder,
            "camera_temperature.jsonl"
        )

        with open(temperature_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        logger.info(
            "Temperature upload completed | date=%s | readings_count=%d | file_path=%s",
            date_folder,
            len(readings),
            temperature_path
        )

        return jsonify({"status": "ok"}), 200

    except Exception:
        logger.exception("Temperature upload failed due to unexpected error")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Receiver server starting | host=0.0.0.0 | port=5000")

    serve(
        app,
        host="0.0.0.0",
        port=5000
    )
