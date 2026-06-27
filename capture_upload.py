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

MANUAL_IMAGE_MODE = "manual"
AUTO_SHARP_IMAGE_MODE = "auto_sharp"

# Basler ace acA5472-5gm sharp auto preset.
# Keep auto exposure short to reduce motion blur on coil defects. If images are
# still dark at these values, add light before increasing exposure too much.
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

# Basler ace GigE transport/acquisition guardrails. These are best-effort:
# unsupported nodes are ignored so the same script can run across pylon/SFNC
# versions.
BASLER_GIGE_DEFAULTS = {
    "packet_size": 1500,
    "inter_packet_delay": 1000,
    "frame_transmission_delay": 0,
    "heartbeat_timeout": 3000,
    "acquisition_frame_rate": 5.0,
}

# Each camera has its own exposure, gain, device selector, and capture delays.
# delay_after_previous is cumulative per camera:
# CAM1 CAP1 at 10s and CAM1 CAP2 at 16s with the default values below.
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

SAVE_DIR = Path.home() / "coil_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_TEMP_LOG_FILE = SAVE_DIR / "camera_temp.log"

PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
UPLOAD_RETRY_DELAY = 2

upload_queue = queue.Queue()
coil_counter = 1
coil_counter_date = datetime.now().date()


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


def start_next_coil():
    global coil_counter, coil_counter_date

    started_at = datetime.now()
    current_date = started_at.date()

    if coil_counter_date != current_date:
        coil_counter = 1
        coil_counter_date = current_date
        log(f"New date {current_date.isoformat()}: coil counter reset to 1")

    coil = f"COIL_{coil_counter}"
    return coil, started_at


def save_camera_temp_log(readings):
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} | " + ", ".join(readings)

    with open(CAMERA_TEMP_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================
# CAMERA
# =============================

def node_names(names):
    if isinstance(names, str):
        return (names,)

    return names


def get_camera_node(cam, names):
    for name in node_names(names):
        try:
            return getattr(cam, name)
        except Exception:
            continue

    return None


def read_camera_node(cam, names):
    node = get_camera_node(cam, names)

    if node is None:
        return None

    try:
        return node.GetValue()
    except Exception:
        try:
            return node.Value
        except Exception:
            return None


def coerce_camera_node_value(node, value):
    try:
        minimum = node.GetMin()
        maximum = node.GetMax()
        value = max(minimum, min(maximum, value))
    except Exception:
        pass

    try:
        current = node.GetValue()
        if isinstance(current, int) and not isinstance(current, bool):
            try:
                increment = node.GetInc()
                minimum = node.GetMin()

                if increment:
                    value = minimum + round((value - minimum) / increment) * increment
            except Exception:
                pass

            return int(round(value))
    except Exception:
        pass

    return value


def set_camera_node(cam, names, value):
    if value is None:
        return False

    for name in node_names(names):
        try:
            node = getattr(cam, name)
            node.SetValue(coerce_camera_node_value(node, value))
            return True
        except Exception:
            continue

    return False


def set_camera_node_fraction(cam, names, fraction):
    fraction = max(0.0, min(1.0, fraction))

    for name in node_names(names):
        try:
            node = getattr(cam, name)
            minimum = node.GetMin()
            maximum = node.GetMax()
            value = minimum + ((maximum - minimum) * fraction)
            node.SetValue(coerce_camera_node_value(node, value))
            return True
        except Exception:
            continue

    return False


def set_auto_target_brightness(cam, target_brightness):
    for name in (
        "AutoTargetBrightness",
        "AutoTargetValue",
        "AutoTargetValueRaw",
    ):
        try:
            node = getattr(cam, name)
            minimum = node.GetMin()
            maximum = node.GetMax()
            value = target_brightness

            if maximum <= 1.0 and value > 1.0:
                value = value / 255.0
            elif maximum > 255.0 and value <= 255.0:
                value = minimum + ((maximum - minimum) * (value / 255.0))

            node.SetValue(coerce_camera_node_value(node, value))
            return True
        except Exception:
            continue

    return False


def set_auto_function_profile_for_sharpness(cam):
    node = get_camera_node(cam, "AutoFunctionProfile")

    if node is None:
        return False

    try:
        for symbol in node.GetSymbolics():
            normalized = symbol.lower()

            if "exposure" in normalized and (
                "min" in normalized or "short" in normalized
            ):
                node.SetValue(symbol)
                return True
    except Exception:
        pass

    for symbol in (
        "ExposureMinimum",
        "ExposureTimeMinimum",
        "MinimizeExposureTime",
    ):
        try:
            node.SetValue(symbol)
            return True
        except Exception:
            continue

    return False


def configure_auto_function_roi(cam):
    width = read_camera_node(cam, "Width")
    height = read_camera_node(cam, "Height")

    for selector, brightness_node, offset_x, offset_y, roi_width, roi_height in (
        (
            "AutoFunctionROISelector",
            "AutoFunctionROIUseBrightness",
            "AutoFunctionROIOffsetX",
            "AutoFunctionROIOffsetY",
            "AutoFunctionROIWidth",
            "AutoFunctionROIHeight",
        ),
        (
            "AutoFunctionAOISelector",
            "AutoFunctionAOIUsageIntensity",
            "AutoFunctionAOIOffsetX",
            "AutoFunctionAOIOffsetY",
            "AutoFunctionAOIWidth",
            "AutoFunctionAOIHeight",
        ),
    ):
        set_camera_node(cam, selector, "ROI1")
        set_camera_node(cam, selector, "AOI1")
        set_camera_node(cam, brightness_node, True)
        set_camera_node(cam, offset_x, 0)
        set_camera_node(cam, offset_y, 0)

        if width is not None:
            set_camera_node(cam, roi_width, width)

        if height is not None:
            set_camera_node(cam, roi_height, height)


def get_auto_sharp_settings(config):
    settings = dict(AUTO_SHARP_DEFAULTS)
    settings.update(config.get("auto_sharp_settings", {}))
    return settings


def get_basler_gige_settings(config):
    settings = dict(BASLER_GIGE_DEFAULTS)
    settings.update(config.get("gige_settings", {}))
    return settings


def configure_basler_gige_camera(cam, config):
    settings = get_basler_gige_settings(config)

    set_camera_node(cam, "GevSCPSPacketSize", settings["packet_size"])
    set_camera_node(cam, "GevSCPD", settings["inter_packet_delay"])
    set_camera_node(cam, "GevSCFTD", settings["frame_transmission_delay"])
    set_camera_node(cam, "GevHeartbeatTimeout", settings["heartbeat_timeout"])

    if set_camera_node(cam, "AcquisitionFrameRateEnable", True):
        set_camera_node(
            cam,
            ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
            settings["acquisition_frame_rate"],
        )


def configure_manual_camera(cam, config):
    set_camera_node(cam, "ExposureAuto", "Off")

    if not set_camera_node(
        cam,
        ("ExposureTime", "ExposureTimeAbs"),
        config["exposure_time"],
    ):
        log(f"{config['name']}: exposure time could not be set")

    set_camera_node(cam, "GainAuto", "Off")

    if not set_camera_node(cam, ("Gain", "GainRaw"), config["gain_value"]):
        log(f"{config['name']}: gain could not be set")


def configure_auto_sharp_camera(cam, config):
    settings = get_auto_sharp_settings(config)

    set_camera_node(cam, "ExposureMode", "Timed")
    set_camera_node(cam, "GammaEnable", False)
    set_camera_node(cam, "Gamma", settings["gamma"])
    set_camera_node(cam, ("BlackLevel", "BlackLevelRaw"), settings["black_level"])

    set_camera_node_fraction(
        cam,
        ("NoiseReduction", "BslNoiseReduction"),
        settings["noise_reduction_fraction"],
    )
    set_camera_node_fraction(
        cam,
        ("SharpnessEnhancement", "BslSharpnessEnhancement"),
        settings["sharpness_enhancement_fraction"],
    )

    configure_auto_function_roi(cam)
    set_auto_function_profile_for_sharpness(cam)

    set_camera_node(
        cam,
        ("AutoExposureTimeLowerLimit", "AutoExposureTimeAbsLowerLimit"),
        settings["exposure_time_lower_limit"],
    )
    set_camera_node(
        cam,
        ("AutoExposureTimeUpperLimit", "AutoExposureTimeAbsUpperLimit"),
        settings["exposure_time_upper_limit"],
    )

    if set_camera_node(cam, "AutoGainLowerLimit", settings["gain_lower_limit"]):
        set_camera_node(cam, "AutoGainUpperLimit", settings["gain_upper_limit"])
    else:
        set_camera_node_fraction(cam, "AutoGainRawLowerLimit", 0.0)
        set_camera_node_fraction(
            cam,
            "AutoGainRawUpperLimit",
            settings["gain_raw_upper_fraction"],
        )

    set_auto_target_brightness(cam, settings["target_brightness"])
    set_camera_node(cam, "ExposureAuto", "Continuous")
    set_camera_node(cam, "GainAuto", "Continuous")


def configure_camera(cam, config):
    cam.Open()
    cam.AcquisitionMode.SetValue("Continuous")
    configure_basler_gige_camera(cam, config)

    image_mode = config.get("image_mode", MANUAL_IMAGE_MODE)

    if image_mode == AUTO_SHARP_IMAGE_MODE:
        configure_auto_sharp_camera(cam, config)
    else:
        configure_manual_camera(cam, config)

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


def find_camera_device(devices, config, log_missing=True):
    serial_number = config.get("serial_number")

    if serial_number:
        for device in devices:
            try:
                if device.GetSerialNumber() == serial_number:
                    return device
            except Exception:
                continue

        if log_missing:
            log(f"{config['name']}: serial number {serial_number} not found")

        return None

    device_index = config.get("device_index", 0)

    if device_index < 0 or device_index >= len(devices):
        if log_missing:
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

    log(f"{config['name']}: camera opened ({get_device_label(device)})")
    return cam


def log_configured_camera_detection_status():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        log("NO BASLER CAMERAS FOUND")
        return

    for config in CAMERA_CONFIGS:
        camera_name = config["name"]
        device = find_camera_device(devices, config)

        if device is None:
            log(f"{camera_name}: camera not detected; system will continue")
            continue

        log(
            f"{camera_name}: camera detected/idle "
            f"({get_device_label(device)})"
        )


def read_camera_temperature(camera):
    # Basler ace classic models expose TemperatureAbs. Newer SFNC models often
    # expose DeviceTemperature, optionally behind a DeviceTemperatureSelector.
    for selector in ("Sensor", "Coreboard", "Mainboard", "Camera"):
        set_camera_node(camera, "DeviceTemperatureSelector", selector)
        temperature = read_camera_node(camera, "DeviceTemperature")

        if temperature is not None:
            return temperature

    temperature = read_camera_node(camera, "TemperatureAbs")

    if temperature is not None:
        return temperature

    raise RuntimeError("temperature node not available")


def release_grab_result(result):
    if result is None:
        return

    try:
        result.Release()
    except Exception:
        pass


def wait_for_frame_trigger_ready(cam, timeout_ms=5000):
    try:
        cam.WaitForFrameTriggerReady(timeout_ms, pylon.TimeoutHandling_ThrowException)
        return True
    except Exception:
        return False


def execute_software_trigger(cam, wait=True):
    if wait:
        wait_for_frame_trigger_ready(cam)

    cam.ExecuteSoftwareTrigger()


def discard_auto_settle_frames(cam, camera_config, capture_name):
    if camera_config.get("image_mode") != AUTO_SHARP_IMAGE_MODE:
        return

    settings = get_auto_sharp_settings(camera_config)
    settle_frames = int(settings.get("auto_settle_frames", 0))
    camera_name = camera_config["name"]

    for frame_no in range(settle_frames):
        result = None

        try:
            execute_software_trigger(cam)
            result = cam.RetrieveResult(5000)

            if not result.GrabSucceeded():
                log(
                    f"{camera_name} {capture_name}: "
                    f"auto settle frame {frame_no + 1} failed"
                )
                return
        except Exception as e:
            log(
                f"{camera_name} {capture_name}: "
                f"auto settle frame {frame_no + 1} error: {e}"
            )
            return
        finally:
            release_grab_result(result)


def read_capture_settings(cam, camera_config):
    settings = {
        "image_mode": camera_config.get("image_mode", MANUAL_IMAGE_MODE),
        "exposure_auto": read_camera_node(cam, "ExposureAuto"),
        "exposure_time": read_camera_node(
            cam,
            ("ExposureTime", "ExposureTimeAbs"),
        ),
        "gain_auto": read_camera_node(cam, "GainAuto"),
        "gain": read_camera_node(cam, ("Gain", "GainRaw")),
        "auto_target_brightness": read_camera_node(
            cam,
            ("AutoTargetBrightness", "AutoTargetValue", "AutoTargetValueRaw"),
        ),
        "gige_packet_size": read_camera_node(cam, "GevSCPSPacketSize"),
        "gige_inter_packet_delay": read_camera_node(cam, "GevSCPD"),
        "gige_frame_transmission_delay": read_camera_node(cam, "GevSCFTD"),
        "gige_heartbeat_timeout": read_camera_node(cam, "GevHeartbeatTimeout"),
        "acquisition_frame_rate": read_camera_node(
            cam,
            ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
        ),
    }

    try:
        settings["camera_temperature_c"] = read_camera_temperature(cam)
    except Exception:
        settings["camera_temperature_c"] = None

    return settings


def temperature_worker(cameras, camera_lock):
    while True:
        readings = []

        with camera_lock:
            for config in CAMERA_CONFIGS:
                camera_name = config["name"]
                cam = cameras.get(camera_name)

                if cam is None:
                    readings.append(f"{camera_name}=idle/off for cooling")
                    continue

                try:
                    temperature = read_camera_temperature(cam)
                    readings.append(f"{camera_name}={temperature:.1f} C")
                except Exception as e:
                    readings.append(f"{camera_name}=temperature unavailable ({e})")

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
    captured_at=None,
    trigger_now=True,
):
    res = None
    img = None
    camera_name = camera_config["name"]
    capture_name = capture_config["name"]

    try:
        if captured_at is None:
            captured_at = datetime.now()

        if trigger_now:
            discard_auto_settle_frames(cam, camera_config, capture_name)
            execute_software_trigger(cam)
            captured_at = datetime.now()

        res = cam.RetrieveResult(5000)

        if not res.GrabSucceeded():
            log(f"{camera_name} {capture_name}: grab failed")
            return False

        date_folder = coil_started_at.strftime("%Y-%m-%d")
        folder = SAVE_DIR / date_folder / coil_folder
        folder.mkdir(parents=True, exist_ok=True)

        timestamp = captured_at.strftime("%H%M%S")
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
            "captured_at": captured_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
        }
        metadata.update(read_capture_settings(cam, camera_config))

        upload_queue.put((path, metadata))
        log(
            f"{camera_name} {capture_name}: captured @ "
            f"{captured_at.strftime('%H:%M:%S')}, saved and queued "
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
            release_grab_result(res)


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


def build_capture_groups():
    groups = []

    for item in build_capture_schedule():
        if not groups or groups[-1]["capture_at"] != item["capture_at"]:
            groups.append({"capture_at": item["capture_at"], "items": []})

        groups[-1]["items"].append(item)

    return groups


def format_capture_group_label(group):
    return " + ".join(
        f"{item['camera_config']['name']} {item['capture_config']['name']}"
        for item in group["items"]
    )


def wait_until_capture(start_time, capture_at, label):
    while True:
        elapsed = time.monotonic() - start_time
        remaining = capture_at - elapsed

        if remaining <= 0:
            return

        log(f"{label} in {int(remaining + 0.999)}s")
        time.sleep(min(1, remaining))


def prepare_capture_group(group, cameras):
    active_captures = []

    for item in group["items"]:
        camera_config = item["camera_config"]
        capture_config = item["capture_config"]
        camera_name = camera_config["name"]
        capture_name = capture_config["name"]
        label = f"{camera_name} {capture_name}"

        cam = open_camera(camera_config)
        cameras[camera_name] = cam

        if cam is None:
            log(f"{label}: skipped because camera is unavailable")
            continue

        discard_auto_settle_frames(cam, camera_config, capture_name)
        active_captures.append(
            {
                "cam": cam,
                "camera_config": camera_config,
                "capture_config": capture_config,
                "label": label,
                "triggered": False,
                "captured_at": None,
            }
        )

    return active_captures


def send_group_triggers(active_captures):
    for active_capture in active_captures:
        if not wait_for_frame_trigger_ready(active_capture["cam"], 500):
            log(f"{active_capture['label']}: trigger-ready wait timed out")

    captured_at = datetime.now()

    for active_capture in active_captures:
        try:
            execute_software_trigger(active_capture["cam"], wait=False)
            active_capture["triggered"] = True
            active_capture["captured_at"] = captured_at
            log(
                f"{active_capture['label']}: software trigger sent @ "
                f"{captured_at.strftime('%H:%M:%S')}"
            )
        except Exception as e:
            log(f"{active_capture['label']}: software trigger failed: {e}")


def save_group_captures(active_captures, coil, coil_folder, coil_started_at):
    for active_capture in active_captures:
        if not active_capture["triggered"]:
            continue

        if not capture_image(
            active_capture["cam"],
            coil,
            coil_folder,
            coil_started_at,
            active_capture["camera_config"],
            active_capture["capture_config"],
            captured_at=active_capture["captured_at"],
            trigger_now=False,
        ):
            log(f"{active_capture['label']}: capture did not complete")


def close_capture_group(cameras, active_captures):
    for active_capture in active_captures:
        camera_name = active_capture["camera_config"]["name"]
        close_camera(active_capture["cam"])
        cameras[camera_name] = None
        log(f"{camera_name}: stopped and closed for idle cooling")


def process_capture_group(
    group,
    cameras,
    camera_lock,
    start_time,
    coil,
    coil_folder,
    coil_started_at,
):
    capture_at = group["capture_at"]
    group_label = format_capture_group_label(group)
    prepare_at = max(0, capture_at - CAMERA_CAPTURE_PREPARE_SECONDS)
    active_captures = []

    log(f"{group_label} due at +{capture_at}s (group trigger)")
    wait_until_capture(start_time, prepare_at, f"{group_label} prepare")

    with camera_lock:
        try:
            log(f"{group_label}: opening cameras for grouped capture")
            active_captures = prepare_capture_group(group, cameras)

            if not active_captures:
                return

            wait_until_capture(start_time, capture_at, f"{group_label} trigger")

            late_by = time.monotonic() - start_time - capture_at
            if late_by > 0.2:
                log(
                    f"{group_label}: trigger late by {late_by:.1f}s; "
                    "increase CAMERA_CAPTURE_PREPARE_SECONDS if this repeats"
                )

            log(f"{group_label}: triggering now")
            send_group_triggers(active_captures)
            save_group_captures(
                active_captures,
                coil,
                coil_folder,
                coil_started_at,
            )
        finally:
            close_capture_group(cameras, active_captures)


def process_coil(cameras, camera_lock):
    global coil_counter

    coil, coil_started_at = start_next_coil()
    coil_folder = build_coil_folder_name(coil, coil_started_at)
    capture_groups = build_capture_groups()
    start_time = time.monotonic()

    log(f"START {coil} -> {coil_folder}")

    for group in capture_groups:
        process_capture_group(
            group,
            cameras,
            camera_lock,
            start_time,
            coil,
            coil_folder,
            coil_started_at,
        )

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
            log_configured_camera_detection_status()

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
