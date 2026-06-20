# SBM Coil Image Capture and Receiver

This project contains two Python scripts for capturing coil images on one Raspberry Pi and receiving them on another system over the local network.

## Files

| File | Purpose |
| --- | --- |
| `capture_upload.py` | Runs on the camera/sensor Raspberry Pi. It waits for a GPIO signal, captures two Basler camera images per coil, queues them, uploads them to the receiver, and deletes local copies after a successful upload. |
| `receiver.py` | Runs on the receiving Raspberry Pi or server. It exposes a Flask upload endpoint, saves each image, and writes matching metadata as JSON. |

## Workflow

1. The photoelectric sensor is connected to GPIO `16`.
2. `capture_upload.py` confirms the sensor is HIGH for `2` seconds.
3. For each confirmed coil, the camera captures:
   - `CAP1` after `10` seconds
   - `CAP2` after `6` more seconds
4. Images are saved temporarily under `~/coil_images/YYYY-MM-DD/COIL_N/`.
5. Each image is uploaded to the receiver at `http://192.168.0.106:5000/upload`.
6. `receiver.py` stores images under `received_images/YYYY-MM-DD/COIL_N/`.
7. Metadata for each image is saved as a matching `.json` file.
8. After upload succeeds, the sender deletes its local image copy.

## Requirements

### Sender Raspberry Pi

- Python 3
- Basler camera
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

Update these values in `capture_upload.py` if your hardware or network changes:

```python
GPIO_PIN = 16
EXPOSURE_TIME = 500000.0
GAIN_VALUE = 10.0
CAP1_DELAY = 10
CAP2_DELAY = 6
HIGH_CONFIRM_TIME = 2
LOW_CONFIRM_TIME = 5
PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
```

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
    COIL_1/
      COIL_1_CAP1_HHMMSS.bmp
      COIL_1_CAP1_HHMMSS.json
      COIL_1_CAP2_HHMMSS.bmp
      COIL_1_CAP2_HHMMSS.json
```

Sender temporary output:

```text
~/coil_images/
  YYYY-MM-DD/
    COIL_1/
      COIL_1_CAP1_HHMMSS.bmp
      COIL_1_CAP2_HHMMSS.bmp
```

## Troubleshooting

- If the sender prints `NO CAMERA FOUND`, check the Basler camera connection, Pylon installation, and camera permissions.
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
