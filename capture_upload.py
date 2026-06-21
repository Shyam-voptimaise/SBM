import time
from datetime import datetime
from pathlib import Path
import requests
import json
import threading
import queue

from gpiozero import Device, DigitalInputDevice
from gpiozero.pins.lgpio import LGPIOFactory
Device.pin_factory = LGPIOFactory()

from pypylon import pylon


# =============================
# CONFIG
# =============================

GPIO_PIN = 16
upload_queue = queue.Queue()
EXPOSURE_TIME = 300000.0
GAIN_VALUE = 10.0

CAP1_DELAY = 10
CAP2_DELAY = 6

HIGH_CONFIRM_TIME = 2   # Sensor must stay HIGH for 2 sec
LOW_CONFIRM_TIME = 5   # Sensor must stay LOW for 5 sec

SAVE_DIR = Path.home() / "coil_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PI2_UPLOAD_URL = "http://192.168.0.106:5000/upload"
UPLOAD_RETRIES = 3

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
# =============================
# CAMERA
# =============================

def configure_camera(cam):
    cam.Open()
    cam.AcquisitionMode.SetValue("Continuous")

    cam.ExposureAuto.SetValue("Off")
    try:
        cam.ExposureTime.SetValue(EXPOSURE_TIME)
    except:
        cam.ExposureTimeAbs.SetValue(EXPOSURE_TIME)

    try:
        cam.GainAuto.SetValue("Off")
        cam.Gain.SetValue(GAIN_VALUE)
    except:
        try:
            cam.GainRaw.SetValue(int(GAIN_VALUE))
        except:
            pass

    cam.TriggerMode.SetValue("On")
    cam.TriggerSource.SetValue("Software")
    cam.TriggerSelector.SetValue("FrameStart")

    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)


def open_camera():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        print("❌ NO CAMERA FOUND")
        return None

    cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    configure_camera(cam)

    print("📷 CAMERA READY")
    return cam


# =============================
# CAPTURE
# =============================

def capture(cam, coil, cap):

    try:

        cam.ExecuteSoftwareTrigger()

        res = cam.RetrieveResult(5000)

        if not res.GrabSucceeded():

            print("❌ GRAB FAILED")

            res.Release()

            return False

        date_folder = datetime.now().strftime(
            "%Y-%m-%d"
        )

        folder = (
            SAVE_DIR
            / date_folder
            / coil
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%H%M%S"
        )

        filename = (
            f"{coil}_{cap}_{timestamp}.bmp"
        )

        path = folder / filename

        img = pylon.PylonImage()

        img.AttachGrabResultBuffer(res)

        img.Save(
            pylon.ImageFileFormat_Bmp,
            str(path)
        )

        img.Release()

        res.Release()

        print(
            f"📸 {cap} SAVED → {filename}"
        )

        upload_queue.put(
            (path, coil)
        )

        print(
            f" Queued for upload → "
            f"{filename}"
        )
        return True

    except Exception as e:

        print(
            f"❌ CAMERA ERROR: {e}"
        )

        return False
# =============================
# PROCESS COIL
# =============================

def process_coil(cam):
    global coil_counter

    coil = f"COIL_{coil_counter}"

    log(f"🚀 START {coil}")
    log(f"⏳ CAP1 will capture after {CAP1_DELAY} seconds")

    for i in range(CAP1_DELAY, 0, -1):
        log(f"⏳ CAP1 in {i}s")
        time.sleep(1)

    log("📷 CAP1 capture starting now")

    if not capture(cam, coil, "CAP1"):
        cam = open_camera()

    log(f"⏳ CAP2 will capture after {CAP2_DELAY} seconds")

    for i in range(CAP2_DELAY, 0, -1):
        log(f"⏳ CAP2 in {i}s")
        time.sleep(1)

    log("📷 CAP2 capture starting now")

    if not capture(cam, coil, "CAP2"):
        cam = open_camera()

    log("✅ PROCESS COMPLETE")

    coil_counter += 1
    return cam


# =============================
# UPLOAD TO PI2
# =============================
def upload_file(file_path, coil_no):

    metadata = {
        "coil_no": coil_no,
        "timestamp": datetime.now().isoformat()
    }

    for attempt in range(UPLOAD_RETRIES):

        try:

            with open(file_path, "rb") as f:

                files = {
                    "image": (
                        file_path.name,
                        f,
                        "image/bmp"
                    )
                }

                data = {
                    "metadata": json.dumps(metadata)
                }

                response = requests.post(
                    PI2_UPLOAD_URL,
                    files=files,
                    data=data,
                    timeout=15
                )

            if response.status_code == 200:

                print(
                    f"☁️ Uploaded → {file_path.name}"
                )

                return True

            print(
                f"❌ Upload failed ({response.status_code})"
            )

        except Exception as e:

            print(
                f"⚠️ Upload attempt {attempt+1} failed: {e}"
            )

        time.sleep(2)

    return False


def uploader_worker():

    while True:

        file_path, coil_no = upload_queue.get()

        try:

            success = upload_file(
                file_path,
                coil_no
            )

            if success:

                try:

                    file_path.unlink()

                    print(
                        f"🗑️ Deleted local file: "
                        f"{file_path.name}"
                    )

                except Exception as e:

                    print(
                        f"⚠️ Delete failed: {e}"
                    )

        except Exception as e:

            print(
                f"⚠️ Upload worker error: {e}"
            )

        upload_queue.task_done()

# =============================
# MAIN
# =============================

def main():

    cam = None
    while cam is None:
        cam = open_camera()
        time.sleep(2)
    threading.Thread(target=uploader_worker, daemon=True).start()

    STATE_IDLE = 0
    STATE_CONFIRM_HIGH = 1
    STATE_WAIT_LOW = 2

    state = STATE_IDLE

    high_start = None
    low_start = None

    print("\n🔥 SYSTEM READY\n")

    while True:

        curr = trigger.value
        now = time.time()

        # =====================================
        # STATE_IDLE
        # Waiting for possible coil
        # =====================================

        if state == STATE_IDLE:

            if curr:

                high_start = now
                state = STATE_CONFIRM_HIGH

                print("\n🟡 HIGH DETECTED - CONFIRMING")

            time.sleep(0.05)
            continue

        # =====================================
        # STATE_CONFIRM_HIGH
        # Require HIGH continuously for 2 sec
        # =====================================

        if state == STATE_CONFIRM_HIGH:

            if curr:

                elapsed = now - high_start

                print(
                    f"⏳ HIGH CONFIRM "
                    f"{int(elapsed)}/{HIGH_CONFIRM_TIME}s",
                    end="\r"
                )

                if elapsed >= HIGH_CONFIRM_TIME:

                    print(
                        f"\n🔵 COIL CONFIRMED @ "
                        f"{datetime.now().strftime('%H:%M:%S')}"
                    )

                    cam = process_coil(cam)

                    state = STATE_WAIT_LOW
                    low_start = None

                    print(
                        f"\n🔒 WAITING FOR LOW "
                        f"{LOW_CONFIRM_TIME}s"
                    )

            else:

                print("\n❌ HIGH INTERRUPTED")

                state = STATE_IDLE
                high_start = None

            time.sleep(0.05)
            continue

        # =====================================
        # STATE_WAIT_LOW
        # Require LOW continuously for 10 sec
        # =====================================

        if state == STATE_WAIT_LOW:

            if not curr:

                if low_start is None:

                    low_start = now

                    print(
                        "\n⚠️ LOW DETECTED - CONFIRMING EXIT"
                    )

                elapsed = now - low_start

                print(
                    f"⏳ LOW CONFIRM "
                    f"{int(elapsed)}/{LOW_CONFIRM_TIME}s",
                    end="\r"
                )

                if elapsed >= LOW_CONFIRM_TIME:

                    print("\n✅ COIL EXIT CONFIRMED")
                    print("🔓 SYSTEM READY FOR NEXT COIL\n")

                    state = STATE_IDLE
                    high_start = None
                    low_start = None

            else:

                if low_start is not None:

                    print(
                        "\n⚠️ LOW INTERRUPTED - RESET TIMER"
                    )

                low_start = None

            time.sleep(0.05)
            continue
# =============================
# START
# =============================

if __name__ == "__main__":
    main()
