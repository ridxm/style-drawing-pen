import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# USB-serial pen firmware (Arduino Nano 33 BLE Sense). Leave SERIAL_PORT empty
# to auto-discover any Arduino-like device (vid 9025 / "Nano" in description).
SERIAL_PORT = ""
SERIAL_BAUD = 115200

CAMERA_INDEX = 0
HOST = "0.0.0.0"
PORT = 5001
CAPTURE_RATE = 30
