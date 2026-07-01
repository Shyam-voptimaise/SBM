from flask import Flask, request, jsonify

import os
import json

from datetime import datetime
from waitress import serve

app = Flask(__name__)

BASE_DIR = "received_images"
TEMPERATURE_DIR = "received_temperatures"

os.makedirs(
    BASE_DIR,
    exist_ok=True
)

os.makedirs(
    TEMPERATURE_DIR,
    exist_ok=True
)


@app.route("/")
def home():
    return "Receiver Running"


@app.route("/upload", methods=["POST"])
def upload():

    try:

        if "image" not in request.files:
            return jsonify(
                {"error": "No image"}
            ), 400

        image = request.files["image"]

        metadata_str = request.form.get(
            "metadata",
            "{}"
        )

        metadata = json.loads(
            metadata_str
        )

        coil_no = metadata.get(
            "coil_no",
            "UNKNOWN"
        )

        date_folder = datetime.now().strftime(
            "%Y-%m-%d"
        )

        save_folder = os.path.join(
            BASE_DIR,
            date_folder,
            coil_no
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        image_path = os.path.join(
            save_folder,
            image.filename
        )

        image.save(image_path)

        json_name = (
            os.path.splitext(
                image.filename
            )[0]
            + ".json"
        )

        metadata_path = os.path.join(
            save_folder,
            json_name
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        print(
            f" Image Saved : {image_path}"
        )

        print(
            f" Metadata Saved : {metadata_path}"
        )

        return jsonify(
            {"status": "ok"}
        ), 200

    except Exception as e:

        print(f"Error: {e}")

        return jsonify(
            {"error": str(e)}
        ), 500


@app.route("/temperature", methods=["POST"])
def temperature():

    try:

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify(
                {"error": "Invalid JSON payload"}
            ), 400

        readings = payload.get(
            "readings"
        )

        if not isinstance(readings, list):
            return jsonify(
                {"error": "Missing readings list"}
            ), 400

        captured_at = payload.get(
            "captured_at"
        )

        if isinstance(captured_at, str) and len(captured_at) >= 10:
            date_folder = captured_at[:10]
        else:
            date_folder = datetime.now().strftime(
                "%Y-%m-%d"
            )

        save_folder = os.path.join(
            TEMPERATURE_DIR,
            date_folder
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        temperature_path = os.path.join(
            save_folder,
            "camera_temperature.jsonl"
        )

        with open(
            temperature_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(payload)
                + "\n"
            )

        print(
            f" Temperature Saved : {temperature_path}"
        )

        return jsonify(
            {"status": "ok"}
        ), 200

    except Exception as e:

        print(f" Temperature Error: {e}")

        return jsonify(
            {"error": str(e)}
        ), 500


if __name__ == "__main__":

    print(
        " Receiver Server Started"
    )

    serve(
        app,
        host="0.0.0.0",
        port=5000
    )