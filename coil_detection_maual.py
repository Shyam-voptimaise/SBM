import subprocess
from datetime import datetime
import os
import time
from PIL import Image
import numpy as np
import shlex

# ======================= Config =======================
PHOTOS_PER_COIL = 1                 # photos per detection
BASE_SAVE_DIR = os.path.join(os.path.expanduser("~"), "photos")
LOG_FILE = os.path.join(BASE_SAVE_DIR, "capture_log.txt")

# Preview / low-light detection
PREVIEW_IMAGE = "/tmp/preview.jpg"
PREVIEW_SIZE = (640, 480)
BRIGHTNESS_THRESHOLD = 40           # <— tweak for your room; 0..255

# Resolution (Arducam 64MP OwlSight native is 9152×6944)
FULL_RES = (9152, 6944)

# Focus settings
USE_AUTOFOCUS = True                # True => AF on capture; False => manual lens position
LENS_POSITION = 3.5                 # only used if USE_AUTOFOCUS=False

# Day profile (short exposure for speed)
DAY_PROFILE = dict(
    exposure="normal",              # normal/short/long
    iso=None,                       # None -> auto
    shutter=None,                   # us; None -> auto
    gain=None,                      # None -> auto
    denoise="cdn_off",              # cdn_off, off, auto, fast, high
)

# Night profile (longer exposure, higher ISO)
NIGHT_PROFILE = dict(
    exposure="long",
    iso=400,                        # 100–800 is reasonable for this sensor
    shutter=150000,                 # 150 ms
    gain=3.5,                       # extra gain if needed
    denoise="auto",
)
# ======================================================

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def cmd_exists(name):
    return subprocess.call(["which", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def ensure_tools():
    for tool in ("rpicam-still",):
        if not cmd_exists(tool):
            raise RuntimeError(f"Required tool '{tool}' not found in PATH")
ensure_tools()

def capture_preview():
    """Grab a tiny, fast preview for light estimation."""
    w, h = PREVIEW_SIZE
    cmd = [
        "rpicam-still",
        "--width", str(w), "--height", str(h),
        "--timeout", "120",
        "--nopreview",
        "--immediate",
        "-o", PREVIEW_IMAGE,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(PREVIEW_IMAGE)

def is_low_light():
    if not capture_preview():
        log("⚠ Preview capture failed; assuming DAY mode.")
        return False
    try:
        img = Image.open(PREVIEW_IMAGE).convert("L")
        brightness = float(np.array(img).mean())
        log(f"📷 Preview brightness: {brightness:.2f} (threshold {BRIGHTNESS_THRESHOLD})")
        return brightness < BRIGHTNESS_THRESHOLD
    except Exception as e:
        log(f"⚠ Error analyzing preview brightness: {e}; assuming DAY.")
        return False

def build_still_cmd(photo_path, meta_path, night_mode=False):
    w, h = FULL_RES
    profile = NIGHT_PROFILE if night_mode else DAY_PROFILE

    cmd = [
        "rpicam-still",
        "--width", str(w), "--height", str(h),
        "--timeout", "1000",
        "--awb", "auto",
        "--metering", "centre",
        "--sharpness", "1.5",
        "--contrast", "1.0",
        "--denoise", profile["denoise"],
        "--encoding", "jpg",
        "--nopreview",
        "--metadata", meta_path,         # save JSON metadata
        "-o", photo_path
    ]

    # Exposure / ISO / shutter / gain
    if profile["exposure"]:
        cmd += ["--exposure", profile["exposure"]]
    if profile["iso"] is not None:
        cmd += ["--iso", str(profile["iso"])]
    if profile["shutter"] is not None:
        cmd += ["--shutter", str(profile["shutter"])]
    if profile["gain"] is not None:
        cmd += ["--gain", str(profile["gain"])]

    # Focus
    if USE_AUTOFOCUS:
        # Fast and usually accurate: autofocus is triggered right before capture
        cmd += ["--autofocus-on-capture"]
    else:
        cmd += ["--autofocus-mode", "manual", "--lens-position", str(LENS_POSITION)]

    return cmd

def take_photo(photo_path, night_mode=False):
    meta_path = os.path.splitext(photo_path)[0] + ".json"
    cmd = build_still_cmd(photo_path, meta_path, night_mode)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = round(time.time() - start, 2)
    if result.returncode != 0:
        log(f"❌ rpicam-still failed ({duration}s)")
        log("    CMD: " + shlex.join(cmd))
        log("    STDERR: " + (result.stderr.strip() or "<empty>"))
    else:
        log(f"✅ Capture OK in {duration}s → {photo_path}")
        if os.path.exists(meta_path):
            log(f"🧾 Metadata saved → {meta_path}")
    return result.returncode, duration

# ======================= Main =========================
log("⌨ Press Enter to simulate a new coil detection. Ctrl+C to quit.")
coil_count = 0

try:
    while True:
        input("➡  Coil detected (press Enter)… ")
        coil_count += 1

        coil_dir = os.path.join(BASE_SAVE_DIR, f"coil_{coil_count:04d}")
        os.makedirs(coil_dir, exist_ok=True)
        log(f"🌀 Coil {coil_count} → {coil_dir}")

        night_mode = is_low_light()
        log("🌙 Night mode ON" if night_mode else "☀ Day mode ON")

        for i in range(1, PHOTOS_PER_COIL + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_name = f"photo_{i}_{timestamp}.jpg"
            photo_path = os.path.join(coil_dir, photo_name)
            rc, _ = take_photo(photo_path, night_mode)
            if rc != 0:
                log(f"❌ Failed to save {photo_name}")
            else:
                log(f"📁 Saved: {photo_name}")

except KeyboardInterrupt:
    log("🛑 Program exited by user.")
