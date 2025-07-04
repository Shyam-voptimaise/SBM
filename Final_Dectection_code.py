import lgpio
import subprocess
from datetime import datetime
import os
import time

SENSOR_PIN = 22  # GPIO pin connected to sensor
SAVE_DIR = os.path.join(os.path.expanduser("~"), "photos")
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize GPIO chip
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, SENSOR_PIN)

print("Monitoring sensor (GPIO 22)... Press Ctrl+C to exit.")

previous_state = 0
ready_to_capture = True

try:
    while True:
        sensor_state = lgpio.gpio_read(h, SENSOR_PIN)
        print(f"GPIO 22 state: {sensor_state}", end='\r')

        # Rising edge detected: 0 → 1
        if sensor_state == 1 and previous_state == 0 and ready_to_capture:
            print("\nSensor HIGH detected. Capturing 3 photos...")

            for i in range(3):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                photo_name = f"photo_{timestamp}.jpg"
                photo_path = os.path.join(SAVE_DIR, photo_name)

                subprocess.run([
                    "libcamera-still",
                    "--lens-position", "2.0",
                    "-o", photo_path,
                    "--nopreview",
                    "--timeout", "1"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                print(f"Photo {i+1}/3 saved: {photo_name}")
                if i < 2:
                    time.sleep(2)  # Wait 2 seconds between photos

            ready_to_capture = False  # Wait for falling edge to reset

        # Falling edge detected: 1 → 0
        elif sensor_state == 0 and previous_state == 1:
            print("\nSensor LOW detected. Ready for next trigger.")
            ready_to_capture = True

        previous_state = sensor_state
        time.sleep(0.01)  # Short delay to reduce CPU usage

except KeyboardInterrupt:
    lgpio.gpiochip_close(h)
    print("\nGPIO cleaned up. Exiting.")
