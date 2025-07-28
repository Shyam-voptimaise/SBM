import lgpio
import subprocess
from datetime import datetime
import os
import time

# ========== Configuration ==========
SENSOR_PIN = 22
PHOTOS_PER_COIL = 3  # Change this as needed
BASE_SAVE_DIR = os.path.join(os.path.expanduser("~"), "photos")
LOG_FILE = os.path.join(BASE_SAVE_DIR, "capture_log.txt")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
# ===================================

# Initialize GPIO chip
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, SENSOR_PIN)

def log(message):
    """Prints to console and appends to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, "a") as log_file:
        log_file.write(full_message + "\n")

log("Monitoring sensor on GPIO 22. Waiting for coils...")

previous_state = 0
ready_to_capture = True
coil_count = 0

try:
    while True:
        sensor_state = lgpio.gpio_read(h, SENSOR_PIN)

        # Rising edge detected
        if sensor_state == 1 and previous_state == 0 and ready_to_capture:
            coil_count += 1
            coil_dir = os.path.join(BASE_SAVE_DIR, f"coil_{coil_count}")
            os.makedirs(coil_dir, exist_ok=True)

            log(f"Coil {coil_count} detected. Saving to {coil_dir}")

            for photo_index in range(1, PHOTOS_PER_COIL + 1):
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_name = f"photo_{photo_index}_{timestamp_str}.jpg"
                photo_path = os.path.join(coil_dir, photo_name)

                result = subprocess.run([
                    "libcamera-still",
                    "--width", "9152",
                    "--height", "6944",
                    "--quality", "100",
                    "--lens-position", "2.0",
                    "--sharpness", "1",
                    "--contrast", "0",
                    "--brightness", "0",
                    "--saturation", "0",
                    "--exposure", "normal",
                    "--awb", "auto",
                    "--denoise", "cdn_hq",
                    "-o", photo_path,
                    "--preview",
                    "--timeout", "1000"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if result.returncode == 0:
                    log(f"Photo {photo_index}/{PHOTOS_PER_COIL} saved: {photo_name}")
                else:
                    log(f"Error capturing photo {photo_index}/{PHOTOS_PER_COIL}")

                if photo_index < PHOTOS_PER_COIL:
                    time.sleep(3)

            ready_to_capture = False

        # Falling edge: coil leaves
        elif sensor_state == 0 and previous_state == 1:
            log("Coil left. Ready for next detection.")
            ready_to_capture = True

        previous_state = sensor_state
        time.sleep(0.01)

except KeyboardInterrupt:
    lgpio.gpiochip_close(h)
    log("GPIO cleaned up. Program exited by user.")
