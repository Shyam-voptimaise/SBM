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
- Both cameras use day/night profiles by default. The day profile starts at
  `06:00` IST and the night profile starts at `18:00` IST.
- Camera profile changes are checked in a background worker and also enforced
  immediately before each capture, so a capture uses the active IST profile.
- Each camera captures `CAP1` after `10` seconds and `CAP2` after `6` more
  seconds. The schedule is cumulative per camera and sorted across cameras.
- Coil numbering is based on the IST date. It starts at `01` each new IST day
  and continues after restarts using `~/coil_images/.coil_sequence_state.json`
  plus a scan of the current day's saved BMP files.
- Missing cameras do not stop the process; reconnects are attempted while the
  runtime continues.
- Camera temperatures use Basler `TemperatureAbs` readings and are logged every
  `10` seconds by default.
- Camera temperature readings are also queued and uploaded as JSON to the
  receiver's `/temperature` endpoint.
- Local BMP files are deleted only after a successful upload.
- Failed uploads are requeued and retried after the configured delay.

## Configuration

Edit `config/runtime.yaml` for production:

| Section | Key Fields |
| --- | --- |
| `gpio` | Pin, pull mode, debounce, high confirmation, and low confirmation timings. |
| `paths` | Sender image root and optional camera temperature log file. |
| `upload` | Receiver URL, HTTP timeout, and retry delay. |
| `temperature_upload` | Receiver temperature URL, HTTP timeout, and retry delay. |
| `camera_runtime` | Camera reconnect, temperature log, and profile check intervals. |
| `logging` | Console logging, rotating file logging, level, and noisy library suppression. |
| `cameras` | Camera names, device indexes or serial numbers, profiles, exposure/gain fallback, and captures. |

Paths support `~` expansion. Set `paths.camera_temperature_log_file` to `null`
to disable camera temperature logging.

Camera profiles are configured per camera:

```yaml
profiles:
  - name: "day"
    exposure_time: 500000.0
    gain_value: 10.0
    start: "06:00"
  - name: "night"
    exposure_time: 500000.0
    gain_value: 10.0
    start: "18:00"
```

The active profile is the latest profile whose `start` time has passed in IST.
For example, `night` remains active after midnight until `day` starts again.

## Sender Output

BMP images are written locally before upload:

```text
~/coil_images/
  YYYY-MM-DD/
    COIL_YYYYMMDD_HHMMSS_01/
      cam_01_cap_01_coil_01.bmp
      cam_01_cap_02_coil_01.bmp
      cam_02_cap_01_coil_01.bmp
      cam_02_cap_02_coil_01.bmp
```

Image filenames follow `cam_XX_cap_YY_coil_ZZ.bmp`.

Default temperature logging writes to:

```text
~/coil_images/camera_temp.log
```

Application logs are written in a parseable production format:

```text
2026-01-02T03:04:05.123+05:30 | INFO | capture.runtime | pid=1234 | sbm-camera-temperature | runtime.py:131 | Camera temperature: CAM1=37.3 C
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

`uploaded_at` is added immediately before upload.

## Receiver Temperature Contract

The sender posts camera temperatures to the configured temperature URL with:

- HTTP method: `POST`
- content type: JSON
- default endpoint: `/temperature`
- success condition: HTTP status `200`

Example payload:

```json
{
  "captured_at": "2026-07-01T14:21:30.147+05:30",
  "uploaded_at": "2026-07-01T14:21:31.000000",
  "readings": [
    {
      "camera_name": "CAM1",
      "temperature_c": 58.0,
      "status": "ok"
    },
    {
      "camera_name": "CAM2",
      "temperature_c": null,
      "status": "not detected"
    }
  ]
}
```

The bundled receiver appends these payloads to:

```text
received_temperatures/YYYY-MM-DD/camera_temperature.jsonl
```

## Receiver

Start the existing receiver separately:

```bash
uv run python receiver.py
```

It listens on `0.0.0.0:5000` and accepts images at `/upload` and temperature
payloads at `/temperature`.

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
