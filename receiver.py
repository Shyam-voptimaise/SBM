from flask import Flask, request, jsonify

import os
import json

from datetime import datetime
from waitress import serve

app = Flask(__name__)

BASE_DIR = "received_images"

os.makedirs(
    BASE_DIR,
    exist_ok=True
)


def safe_folder_name(value, fallback):
    value = str(value or "").strip()

    if not value:
        return fallback

    safe_value = "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in value
    ).strip("_")

    return safe_value or fallback


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

        date_folder = safe_folder_name(
            metadata.get(
                "coil_date",
                datetime.now().strftime("%Y-%m-%d")
            ),
            datetime.now().strftime("%Y-%m-%d")
        )

        coil_folder = safe_folder_name(
            metadata.get(
                "coil_folder",
                metadata.get(
                    "coil_no",
                    "UNKNOWN"
                )
            ),
            "UNKNOWN"
        )

        save_folder = os.path.join(
            BASE_DIR,
            date_folder,
            coil_folder
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
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        print(
            f"✅ Image Saved : {image_path}"
        )

        print(
            f"✅ Metadata Saved : {metadata_path}"
        )

        return jsonify(
            {"status": "ok"}
        ), 200

    except Exception as e:

        print(f"❌ Error: {e}")

        return jsonify(
            {"error": str(e)}
        ), 500


if __name__ == "__main__":

    print(
        "🚀 Receiver Server Started"
    )

    serve(
        app,
        host="0.0.0.0",
        port=5000
    )
