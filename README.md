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
3. For each confirmed coil, cameras with the same due time are opened briefly, prepared, and software-triggered together:
   - `CAM1 CAP1` after `10` seconds
   - `CAM1 CAP2` after `6` more seconds
   - `CAM2 CAP1` after its own configured delay
   - `CAM2 CAP2` after its own configured second delay
4. After every capture group, the active camera(s) are stopped and closed for idle cooling.
5. If a camera is not detected, that camera is logged and skipped while the system keeps running.
6. Images are saved temporarily under `~/coil_images/YYYY-MM-DD/COIL_YYYYMMDD_HHMMSS_COIL_N/`.
7. Each image is queued for upload to the receiver at `http://192.168.0.106:5000/upload`.
8. Uploads keep retrying in the background until they succeed, even when another coil is triggered.
9. `receiver.py` stores images under `received_images/YYYY-MM-DD/COIL_YYYYMMDD_HHMMSS_COIL_N/`.
10. Metadata for each image is saved as a matching `.json` file.
11. After upload succeeds, the sender deletes its local image copy.

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

Update these values in `capture_upload.py` if your hardware or network changes. Each camera has its own device selector, image mode, exposure/gain fallback, and capture delays:

```python
GPIO_PIN = 16

MANUAL_IMAGE_MODE = "manual"
AUTO_SHARP_IMAGE_MODE = "auto_sharp"

AUTO_SHARP_DEFAULTS = {
    "exposure_time_lower_limit": 500.0,
    "exposure_time_upper_limit": 8000.0,
    "gain_lower_limit": 0.0,
    "gain_upper_limit": 8.0,
    "gain_raw_upper_fraction": 0.35,
    "target_brightness": 110.0,
    "auto_settle_frames": 4,
    "gamma": 1.0,
    "black_level": 0.0,
    "noise_reduction_fraction": 0.0,
    "sharpness_enhancement_fraction": 0.35,
}

BASLER_GIGE_DEFAULTS = {
    "packet_size": 1500,
    "inter_packet_delay": 1000,
    "frame_transmission_delay": 0,
    "heartbeat_timeout": 3000,
    "acquisition_frame_rate": 5.0,
}

CAMERA_CONFIGS = [
    {
        "name": "CAM1",
        "device_index": 0,
        "serial_number": "25343487",
        "image_mode": AUTO_SHARP_IMAGE_MODE,
        "exposure_time": 500000.0,
        "gain_value": 10.0,
        "captures": [
            {"name": "CAP1", "delay_after_previous": 10},
            {"name": "CAP2", "delay_after_previous": 6},
        ],
    },
    {
        "name": "CAM2",
        "device_index": 1,
        "serial_number": "25343513",
        "image_mode": AUTO_SHARP_IMAGE_MODE,
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
TEMPERATURE_LOG_INTERVAL = 30
CAMERA_CAPTURE_PREPARE_SECONDS = 4.0
PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
UPLOAD_RETRY_DELAY = 2
```

`AUTO_SHARP_IMAGE_MODE` is tuned for the Basler ace `acA5472-5gm` when the coil image must stay clear for defect detection. It keeps exposure auto and gain auto enabled, but limits auto exposure to `8000 us` so the camera does not blur moving defects by choosing a long exposure. The script also grabs `auto_settle_frames` unsaved frames before each saved image so Basler's auto exposure/gain can settle while using software trigger mode.

If the image is still dark, improve lighting first. If you must tune in software, raise `target_brightness` slowly or raise `gain_upper_limit`; only increase `exposure_time_upper_limit` after checking that the coil defects are still sharp. To return to the old fixed exposure/gain behavior, set a camera's `image_mode` to `MANUAL_IMAGE_MODE`; then `exposure_time` and `gain_value` are used directly.

`BASLER_GIGE_DEFAULTS` applies conservative GigE transport settings when a camera is opened: packet size, inter-packet delay, frame transmission delay, heartbeat timeout, and acquisition frame-rate cap. Unsupported Basler nodes are ignored automatically.

The sender logs each configured camera as `detected/idle` at startup without opening it. During idle time the temperature log reports `idle/off for cooling`; after the first successful reading, it also shows the last measured temperature and time. Temperature is read only while a camera is already open for capture or metadata.

`CAMERA_CAPTURE_PREPARE_SECONDS` opens and settles cameras shortly before the due time. This keeps idle heat down while allowing CAM1 and CAM2 captures with the same delay to receive software triggers in the same second.

If discovery order changes, set each camera's `serial_number` to the Basler serial number and keep `device_index` as a fallback. The current serials came from the service log: CAM1 `25343487`, CAM2 `25343513`.

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
- If a startup line shows a configured camera is not detected, confirm the serial number, camera connection, and network interface IP.
- If temperature logs say `idle/off for cooling`, that is expected while the cameras are closed between captures. After the next successful camera-open reading, the same line will include the last measured temperature.
- If temperature logs say `temperature unavailable`, confirm your Basler model exposes `TemperatureAbs` or `DeviceTemperature`.
- If CAM1 and CAM2 are still triggered late, increase `CAMERA_CAPTURE_PREPARE_SECONDS` so both cameras have enough time to open and settle before the due timestamp.
- If auto mode saves dark images, add more light or increase `target_brightness`/`gain_upper_limit`; avoid raising `exposure_time_upper_limit` too high because long exposure creates blur on moving coil defects.
- If auto mode captures too late, lower `auto_settle_frames`; the default `4` unsaved frames gives auto exposure/gain time to settle before the saved frame.
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
