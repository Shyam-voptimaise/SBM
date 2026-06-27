# SBM Coil Image Capture and Receiver

This repository contains a Raspberry Pi sender for Basler coil image capture and
a small Flask receiver kept for compatibility. Sender code lives in
`src/capture`; `capture_upload.py` is only a backward-compatible launcher.

## Project Layout

| Path | Purpose |
| --- | --- |
| `src/capture/` | Sender package: config, GPIO, camera capture, upload worker, and runtime loop. |
| `capture_upload.py` | Compatibility launcher for existing deployments. |
| `config/runtime.yaml` | Production runtime defaults. |
| `config/runtime.example.yaml` | Documented config template. |
| `receiver.py` | Existing receiver reference. The sender preserves its upload contract. |
| `tests/` | Pure logic tests that do not require GPIO or Basler hardware. |

## Setup

Use `uv`; `pyproject.toml` is the dependency source of truth.

```bash
uv sync
```

Run the sender with the console entrypoint:

```bash
uv run sbm-capture --config config/runtime.yaml
```

The old command is still supported:

```bash
uv run python capture_upload.py
```

You can also set the config path with:

```bash
SBM_RUNTIME_CONFIG=/path/to/config/runtime.yaml uv run sbm-capture
```

## Runtime Behavior

Production defaults preserve the previous sender behavior:

- GPIO uses `gpiozero` with `lgpio`, pin `16`, `pull_up: false`, and
  `bounce_time_seconds: 0.1`.
- A coil is confirmed only after the signal stays HIGH for `2` seconds.
- The sender waits for LOW for `5` seconds before arming for the next coil.
- Two Basler cameras are configured by default:
  `CAM1` uses `device_index: 0`; `CAM2` uses `device_index: 1`.
- Both cameras use `exposure_time: 500000.0` and `gain_value: 10.0`.
- Each camera captures `CAP1` after `10` seconds and `CAP2` after `6` more
  seconds. The schedule is cumulative per camera and sorted across cameras.
- Missing cameras do not stop the process; reconnects are attempted while the
  runtime continues.
- Local BMP files are deleted only after a successful upload.
- Failed uploads are requeued and retried after the configured delay.

## Configuration

Edit `config/runtime.yaml` for production:

| Section | Key Fields |
| --- | --- |
| `gpio` | Pin, pull mode, debounce, high confirmation, and low confirmation timings. |
| `paths` | Sender image root and optional camera temperature log file. |
| `upload` | Receiver URL, HTTP timeout, and retry delay. |
| `camera_runtime` | Camera reconnect and temperature log intervals. |
| `logging` | Console logging, rotating file logging, level, and noisy library suppression. |
| `cameras` | Camera names, device indexes or serial numbers, exposure, gain, and captures. |

Paths support `~` expansion. Set `paths.camera_temperature_log_file` to `null`
to disable camera temperature logging.

## Sender Output

BMP images are written locally before upload:

```text
~/coil_images/
  YYYY-MM-DD/
    COIL_YYYYMMDD_HHMMSS_COIL_N/
      COIL_N_CAM1_CAP1_HHMMSS.bmp
      COIL_N_CAM1_CAP2_HHMMSS.bmp
      COIL_N_CAM2_CAP1_HHMMSS.bmp
      COIL_N_CAM2_CAP2_HHMMSS.bmp
```

Default temperature logging writes to:

```text
~/coil_images/camera_temp.log
```

## Receiver Upload Contract

The sender posts each BMP to the configured receiver URL with:

- HTTP method: `POST`
- multipart file field: `image`
- file content type: `image/bmp`
- form field: `metadata`
- metadata value: JSON string
- success condition: HTTP status `200`

Metadata includes:

```text
coil_no
coil_folder
coil_started_at
coil_date
camera_name
camera_device_index
camera_serial_number
capture_name
delay_after_previous
captured_at
uploaded_at
```

`uploaded_at` is added immediately before upload. `receiver.py` is not
extended by the sender refactor.

## Receiver

Start the existing receiver separately:

```bash
uv run python receiver.py
```

It listens on `0.0.0.0:5000` and accepts uploads at `/upload`.

## Production Migration Notes

1. Back up the current production `config/runtime.yaml`.
2. Run `uv sync` on the Raspberry Pi.
3. Confirm the receiver is running and reachable from the sender.
4. Verify Basler Pylon SDK installation, camera permissions, and USB/network
   camera visibility.
5. Verify GPIO wiring, pin `16`, and receiver URL in `config/runtime.yaml`.
6. Start the sender with `uv run sbm-capture --config config/runtime.yaml`.

## Development Checks

```bash
uv run python -m compileall src capture_upload.py receiver.py
uv run pytest
uv run ruff check .
```
