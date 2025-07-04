from pathlib import Path

readme_content = """# Photo Capture with Photoelectric Sensor and Raspberry Pi

This Python script captures **3 photos** using a Raspberry Pi camera **each time a photoelectric sensor detects a HIGH signal (rising edge)**. Photos are timestamped and saved locally, and the script only triggers again **after the signal drops (falling edge) and rises again**.

---

## 🧰 Requirements

- Raspberry Pi (with Raspberry Pi OS)
- Raspberry Pi Camera Module (enabled)
- Photoelectric Sensor (connected to GPIO)
- Python 3
- `lgpio` library
- `libcamera-still` (comes pre-installed on recent Raspberry Pi OS versions)

---

## 🔌 Hardware Wiring

- Connect your photoelectric sensor's output to **GPIO 22** on the Raspberry Pi.
- Ensure the sensor has proper power (usually 5V and GND).
- The GPIO pin reads **HIGH (1)** when the object is detected.

---

## 📦 Installation

Libraries
sudo apt update
sudo apt install python3-lgpio


