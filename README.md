# SBM Coil Image Capture and Receiver

This project contains two Python scripts for capturing coil images on one Raspberry Pi and receiving them on another system over the local network.

## Files

| File | Purpose |
| --- | --- |
| `capture_upload.py` | Runs on the camera/sensor Raspberry Pi. It waits for a GPIO signal, captures two images from each configured Basler camera per coil, queues them, uploads them to the receiver, and deletes local copies after a successful upload. |
| `receiver.py` | Runs on the receiving Raspberry Pi or server. It exposes a Flask upload endpoint, saves each image, and writes matching metadata as JSON. |

## Workflow

1. The photoelectric sensor is connected to GPIO `16`.
2. `capture_upload.py` confirms the sensor is HIGH for `2` seconds.
3. For each confirmed coil, each detected configured camera captures two images:
   - `CAM1 CAP1` after `10` seconds
   - `CAM1 CAP2` after `6` more seconds
   - `CAM2 CAP1` after its own configured delay
   - `CAM2 CAP2` after its own configured second delay
4. If a camera is not detected, that camera is logged and skipped while the system keeps running.
5. Images are saved temporarily under `~/coil_images/YYYY-MM-DD/COIL_YYYYMMDD_HHMMSS_COIL_N/`.
6. Each image is queued for upload to the receiver at `http://192.168.0.106:5000/upload`.
7. Uploads keep retrying in the background until they succeed, even when another coil is triggered.
8. `receiver.py` stores images under `received_images/YYYY-MM-DD/COIL_YYYYMMDD_HHMMSS_COIL_N/`.
9. Metadata for each image is saved as a matching `.json` file.
10. After upload succeeds, the sender deletes its local image copy.

## Requirements

### Sender Raspberry Pi

- Python 3
- Up to two Basler cameras
- Basler Pylon SDK / pypylon support
- Photoelectric sensor connected to GPIO `16`
- Network access to the receiver
- Python packages:
  - `requests`
  - `gpiozero`
  - `lgpio`
  - `pypylon`

Install packages:

```bash
uv sync
```

### Receiver Raspberry Pi / Server

- Python 3
- Network access from the sender Raspberry Pi
- Python packages:
  - `flask`
  - `waitress`

Install packages:

```bash
uv sync
```

If you prefer syncing directly from `requirements.txt`, run:

```bash
uv pip sync requirements.txt
```

## Configuration

Update these values in `capture_upload.py` if your hardware or network changes. Each camera has its own device selector, exposure, gain, and capture delays:

```python
GPIO_PIN = 16

CAMERA_CONFIGS = [
    {
        "name": "CAM1",
        "device_index": 0,
        "serial_number": None,
        "exposure_time": 300000.0,
        "gain_value": 10.0,
        "captures": [
            {"name": "CAP1", "delay_after_previous": 10},
            {"name": "CAP2", "delay_after_previous": 6},
        ],
    },
    {
        "name": "CAM2",
        "device_index": 1,
        "serial_number": None,
        "exposure_time": 300000.0,
        "gain_value": 10.0,
        "captures": [
            {"name": "CAP1", "delay_after_previous": 10},
            {"name": "CAP2", "delay_after_previous": 6},
        ],
    },
]

HIGH_CONFIRM_TIME = 2
LOW_CONFIRM_TIME = 5
TEMPERATURE_LOG_INTERVAL = 1
CAMERA_RECONNECT_INTERVAL = 5
PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
UPLOAD_RETRY_DELAY = 2
```

If USB discovery order changes, set each camera's `serial_number` to the Basler serial number and keep `device_index` as a fallback. The sender starts even if a configured camera is missing, logs `not detected`, and retries detection every `CAMERA_RECONNECT_INTERVAL` seconds.

In `receiver.py`, the server listens on all network interfaces at port `5000`:

```python
serve(app, host="0.0.0.0", port=5000)
```

## How To Run

Start the receiver first:

```bash
python receiver.py
```

The receiver should print that the server has started. You can also test it in a browser:

```text
http://<receiver-ip>:5000/
```

Then update `PI2_UPLOAD_URL` in `capture_upload.py` with the receiver IP address and start the sender:

```bash
python capture_upload.py
```

## Output Structure

Receiver output:

```text
received_images/
  YYYY-MM-DD/
    COIL_20260622_143052_COIL_1/
      COIL_1_CAM1_CAP1_HHMMSS.bmp
      COIL_1_CAM1_CAP1_HHMMSS.json
      COIL_1_CAM1_CAP2_HHMMSS.bmp
      COIL_1_CAM1_CAP2_HHMMSS.json
      COIL_1_CAM2_CAP1_HHMMSS.bmp
      COIL_1_CAM2_CAP1_HHMMSS.json
      COIL_1_CAM2_CAP2_HHMMSS.bmp
      COIL_1_CAM2_CAP2_HHMMSS.json
```

Sender temporary output:

```text
~/coil_images/
  camera_temp.log
  YYYY-MM-DD/
    COIL_20260622_143052_COIL_1/
      COIL_1_CAM1_CAP1_HHMMSS.bmp
      COIL_1_CAM1_CAP2_HHMMSS.bmp
      COIL_1_CAM2_CAP1_HHMMSS.bmp
      COIL_1_CAM2_CAP2_HHMMSS.bmp
```

## Troubleshooting

- If the sender prints `NO BASLER CAMERAS FOUND`, check both Basler camera connections, Pylon installation, and camera permissions.
- If the sender reports a missing device index, confirm both cameras are connected or set `serial_number` in `CAMERA_CONFIGS`.
- If a temperature line shows `CAM2=not detected`, the system is still running and will retry that camera automatically.
- If temperature logs say `temperature unavailable`, confirm your Basler model exposes `TemperatureAbs`.
- If uploads fail, confirm the receiver is running and that `PI2_UPLOAD_URL` matches the receiver IP address.
- If no coil is detected, verify the sensor wiring, GPIO pin number, and signal level.
- Keep both systems on the same network unless port forwarding or routing is configured.

## Git Upload

Before pushing to GitHub, check the files:

```bash
git status
```

Stage the project:

```bash
git add -A
```

Commit the update:

```bash
git commit -m "Add coil capture receiver project"
```

Push to the configured remote:

```bash
git push origin main
```
