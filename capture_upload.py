import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from gpiozero import Device, DigitalInputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from pypylon import pylon

Device.pin_factory = LGPIOFactory()


# =============================
# CONFIG
# =============================

GPIO_PIN = 16

# Each camera has its own exposure, gain, device selector, and capture delays.
# delay_after_previous is cumulative per camera:
# CAM1 CAP1 at 10s and CAM1 CAP2 at 16s with the default values below.
CAMERA_CONFIGS = [
    {
        "name": "CAM1",
        "device_index": 0,
        "serial_number": None,
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
        "serial_number": None,
        "exposure_time": 500000.0,
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

SAVE_DIR = Path.home() / "coil_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_TEMP_LOG_FILE = SAVE_DIR / "camera_temp.log"

PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
UPLOAD_RETRY_DELAY = 2

upload_queue = queue.Queue()
coil_counter = 1


# =============================
# GPIO
# =============================

trigger = DigitalInputDevice(GPIO_PIN, pull_up=False, bounce_time=0.1)


# =============================
# LOGGING
# =============================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def build_coil_folder_name(coil, started_at):
    return f"COIL_{started_at.strftime('%Y%m%d_%H%M%S')}_{coil}"


def save_camera_temp_log(readings):
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} | " + ", ".join(readings)

    with open(CAMERA_TEMP_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================
# CAMERA
# =============================

def configure_camera(cam, config):
    cam.Open()
    cam.AcquisitionMode.SetValue("Continuous")

    cam.ExposureAuto.SetValue("Off")
    try:
        cam.ExposureTime.SetValue(config["exposure_time"])
    except Exception:
        cam.ExposureTimeAbs.SetValue(config["exposure_time"])

    try:
        cam.GainAuto.SetValue("Off")
        cam.Gain.SetValue(config["gain_value"])
    except Exception:
        try:
            cam.GainRaw.SetValue(int(config["gain_value"]))
        except Exception:
            pass

    cam.TriggerMode.SetValue("On")
    cam.TriggerSource.SetValue("Software")
    cam.TriggerSelector.SetValue("FrameStart")
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)


def get_device_label(device):
    details = []

    try:
        details.append(device.GetModelName())
    except Exception:
        pass

    try:
        details.append(f"SN {device.GetSerialNumber()}")
    except Exception:
        pass

    return " / ".join(details) if details else "UNKNOWN DEVICE"


def find_camera_device(devices, config):
    serial_number = config.get("serial_number")

    if serial_number:
        for device in devices:
            try:
                if device.GetSerialNumber() == serial_number:
                    return device
            except Exception:
                continue

        log(f"{config['name']}: serial number {serial_number} not found")
        return None

    device_index = config.get("device_index", 0)

    if device_index < 0 or device_index >= len(devices):
        log(
            f"{config['name']}: device index {device_index} not found; "
            f"{len(devices)} Basler camera(s) detected"
        )
        return None

    return devices[device_index]


def close_camera(cam):
    if cam is None:
        return

    try:
        if cam.IsGrabbing():
            cam.StopGrabbing()
    except Exception:
        pass

    try:
        if cam.IsOpen():
            cam.Close()
    except Exception:
        pass


def close_all_cameras(cameras):
    for cam in cameras.values():
        close_camera(cam)


def open_camera(config):
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        log("NO BASLER CAMERAS FOUND")
        return None

    device = find_camera_device(devices, config)

    if device is None:
        return None

    cam = pylon.InstantCamera(factory.CreateDevice(device))

    try:
        configure_camera(cam, config)
    except Exception as e:
        log(f"{config['name']}: camera open/config error: {e}")
        close_camera(cam)
        return None

    log(f"{config['name']}: camera ready ({get_device_label(device)})")
    return cam


def open_configured_cameras(cameras=None):
    cameras = cameras or {}

    for config in CAMERA_CONFIGS:
        name = config["name"]

        if cameras.get(name) is not None:
            continue

        cameras[name] = open_camera(config)

    return cameras


def read_camera_temperature(camera):
    # Basler device temperature in Celsius on models that expose TemperatureAbs.
    d = camera.TemperatureAbs.Value
    return d


def log_camera_detection_status(cameras):
    for config in CAMERA_CONFIGS:
        camera_name = config["name"]

        if cameras.get(camera_name) is None:
            log(f"{camera_name}: camera not detected; system will continue")


def temperature_worker(cameras, camera_lock):
    last_reconnect_attempt = time.monotonic()

    while True:
        readings = []
        now = time.monotonic()
        should_reconnect = (
            now - last_reconnect_attempt >= CAMERA_RECONNECT_INTERVAL
        )

        with camera_lock:
            for config in CAMERA_CONFIGS:
                camera_name = config["name"]
                cam = cameras.get(camera_name)

                if cam is None:
                    if should_reconnect:
                        cam = open_camera(config)
                        cameras[camera_name] = cam

                    if cam is None:
                        readings.append(f"{camera_name}=not detected")
                        continue

                try:
                    temperature = read_camera_temperature(cam)
                    readings.append(f"{camera_name}={temperature:.1f} C")
                except Exception as e:
                    readings.append(f"{camera_name}=temperature unavailable ({e})")

        if should_reconnect:
            last_reconnect_attempt = now

        save_camera_temp_log(readings)
        log("Camera temperature: " + ", ".join(readings))
        time.sleep(TEMPERATURE_LOG_INTERVAL)


# =============================
# CAPTURE
# =============================

def capture_image(
    cam,
    coil,
    coil_folder,
    coil_started_at,
    camera_config,
    capture_config,
):
    res = None
    img = None
    camera_name = camera_config["name"]
    capture_name = capture_config["name"]

    try:
        cam.ExecuteSoftwareTrigger()
        res = cam.RetrieveResult(5000)

        if not res.GrabSucceeded():
            log(f"{camera_name} {capture_name}: grab failed")
            return False

        date_folder = coil_started_at.strftime("%Y-%m-%d")
        folder = SAVE_DIR / date_folder / coil_folder
        folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{coil}_{camera_name}_{capture_name}_{timestamp}.bmp"
        path = folder / filename

        img = pylon.PylonImage()
        img.AttachGrabResultBuffer(res)
        img.Save(pylon.ImageFileFormat_Bmp, str(path))

        metadata = {
            "coil_no": coil,
            "coil_folder": coil_folder,
            "coil_started_at": coil_started_at.isoformat(),
            "coil_date": date_folder,
            "camera_name": camera_name,
            "camera_device_index": camera_config.get("device_index"),
            "camera_serial_number": camera_config.get("serial_number"),
            "capture_name": capture_name,
            "delay_after_previous": capture_config["delay_after_previous"],
            "captured_at": datetime.now().isoformat(),
        }

        upload_queue.put((path, metadata))
        log(
            f"{camera_name} {capture_name}: saved and queued "
            f"until upload -> {filename}"
        )
        return True

    except Exception as e:
        log(f"{camera_name} {capture_name}: camera error: {e}")
        return False

    finally:
        if img is not None:
            try:
                img.Release()
            except Exception:
                pass

        if res is not None:
            try:
                res.Release()
            except Exception:
                pass


# =============================
# PROCESS COIL
# =============================

def build_capture_schedule():
    schedule = []

    for camera_config in CAMERA_CONFIGS:
        capture_at = 0

        for capture_config in camera_config["captures"]:
            capture_at += capture_config["delay_after_previous"]
            schedule.append(
                {
                    "capture_at": capture_at,
                    "camera_config": camera_config,
                    "capture_config": capture_config,
                }
            )

    return sorted(schedule, key=lambda item: item["capture_at"])


def wait_until_capture(start_time, capture_at, label):
    while True:
        elapsed = time.monotonic() - start_time
        remaining = capture_at - elapsed

        if remaining <= 0:
            return

        log(f"{label} in {int(remaining + 0.999)}s")
        time.sleep(min(1, remaining))


def process_coil(cameras, camera_lock):
    global coil_counter

    coil = f"COIL_{coil_counter}"
    coil_started_at = datetime.now()
    coil_folder = build_coil_folder_name(coil, coil_started_at)
    schedule = build_capture_schedule()
    start_time = time.monotonic()

    log(f"START {coil} -> {coil_folder}")

    for item in schedule:
        camera_config = item["camera_config"]
        capture_config = item["capture_config"]
        camera_name = camera_config["name"]
        capture_name = capture_config["name"]
        label = f"{camera_name} {capture_name}"

        log(f"{label} scheduled at +{item['capture_at']}s")
        wait_until_capture(start_time, item["capture_at"], label)
        log(f"{label} capture starting now")

        with camera_lock:
            cam = cameras.get(camera_name)

            if cam is None:
                cam = open_camera(camera_config)
                cameras[camera_name] = cam

        if cam is None:
            log(f"{label}: skipped because camera is unavailable")
            continue

        with camera_lock:
            if not capture_image(
                cam,
                coil,
                coil_folder,
                coil_started_at,
                camera_config,
                capture_config,
            ):
                close_camera(cam)
                cameras[camera_name] = open_camera(camera_config)

    log("PROCESS COMPLETE")

    coil_counter += 1
    return cameras


# =============================
# UPLOAD TO PI2
# =============================

def upload_file(file_path, metadata):
    if isinstance(metadata, dict):
        upload_metadata = dict(metadata)
    else:
        upload_metadata = {"coil_no": metadata}

    upload_metadata["uploaded_at"] = datetime.now().isoformat()

    try:
        with open(file_path, "rb") as f:
            files = {
                "image": (
                    file_path.name,
                    f,
                    "image/bmp",
                )
            }

            data = {"metadata": json.dumps(upload_metadata)}

            response = requests.post(
                PI2_UPLOAD_URL,
                files=files,
                data=data,
                timeout=15,
            )

        if response.status_code == 200:
            log(f"Uploaded -> {file_path.name}")
            return True

        log(f"Upload failed ({response.status_code}) -> {file_path.name}")

    except Exception as e:
        log(f"Upload failed for {file_path.name}: {e}")

    return False


def uploader_worker():
    while True:
        file_path, metadata = upload_queue.get()

        try:
            if upload_file(file_path, metadata):
                try:
                    file_path.unlink()
                    log(f"Deleted local file -> {file_path.name}")
                except Exception as e:
                    log(f"Delete failed for {file_path.name}: {e}")
            else:
                upload_queue.put((file_path, metadata))
                log(f"Kept queued for retry -> {file_path.name}")
                time.sleep(UPLOAD_RETRY_DELAY)

        except Exception as e:
            log(f"Upload worker error: {e}")

        upload_queue.task_done()


# =============================
# MAIN
# =============================

def main():
    cameras = {}
    camera_lock = threading.Lock()

    try:
        with camera_lock:
            cameras = open_configured_cameras(cameras)
            log_camera_detection_status(cameras)

        threading.Thread(target=uploader_worker, daemon=True).start()
        threading.Thread(
            target=temperature_worker,
            args=(cameras, camera_lock),
            daemon=True,
        ).start()

        state_idle = 0
        state_confirm_high = 1
        state_wait_low = 2

        state = state_idle
        high_start = None
        low_start = None

        print("\nSYSTEM READY\n")

        while True:
            curr = trigger.value
            now = time.time()

            if state == state_idle:
                if curr:
                    high_start = now
                    state = state_confirm_high
                    print("\nHIGH DETECTED - CONFIRMING")

                time.sleep(0.05)
                continue

            if state == state_confirm_high:
                if curr:
                    elapsed = now - high_start

                    print(
                        f"HIGH CONFIRM {int(elapsed)}/{HIGH_CONFIRM_TIME}s",
                        end="\r",
                    )

                    if elapsed >= HIGH_CONFIRM_TIME:
                        print(
                            f"\nCOIL CONFIRMED @ "
                            f"{datetime.now().strftime('%H:%M:%S')}"
                        )

                        cameras = process_coil(cameras, camera_lock)

                        state = state_wait_low
                        low_start = None

                        print(f"\nWAITING FOR LOW {LOW_CONFIRM_TIME}s")

                else:
                    print("\nHIGH INTERRUPTED")
                    state = state_idle
                    high_start = None

                time.sleep(0.05)
                continue

            if state == state_wait_low:
                if not curr:
                    if low_start is None:
                        low_start = now
                        print("\nLOW DETECTED - CONFIRMING EXIT")

                    elapsed = now - low_start

                    print(
                        f"LOW CONFIRM {int(elapsed)}/{LOW_CONFIRM_TIME}s",
                        end="\r",
                    )

                    if elapsed >= LOW_CONFIRM_TIME:
                        print("\nCOIL EXIT CONFIRMED")
                        print("SYSTEM READY FOR NEXT COIL\n")

                        state = state_idle
                        high_start = None
                        low_start = None

                else:
                    if low_start is not None:
                        print("\nLOW INTERRUPTED - RESET TIMER")

                    low_start = None

                time.sleep(0.05)
                continue

    finally:
        with camera_lock:
            close_all_cameras(cameras)


# =============================
# START
# =============================

if __name__ == "__main__":
    main()
