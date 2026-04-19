"""Serial receiver for the penDNA Arduino firmware over USB CDC.

Firmware (github.com/ridxm/style-transfer-pen, branch Arduino-nano-sense) boots
into calibration, then waits. The pen's physical button toggles recording; while
recording it streams one line per 100 Hz sample:

    roll = 12.34, pitch = -5.67, yaw = 89.01, \\
    fsr0 = 1023, fsr1 = 512, fsr2 = 123, fsr3 = 456, \\
    fsr4 = 12, fsr5 = 45, fsr6 = 78, fsr7 = 99

Status lines start with `#` — boot banner, calibration, gyro bias, and
"--- recording started/stopped ---" markers.

This module parses those lines into the webapp's sample dict
  {t, pressure[4], imu[6], pen_down, roll, pitch, yaw, fsr_raw[N]}
where pressure is the first four FSRs normalized 0..1, and the IMU slots hold
[roll, pitch, yaw, 0, 0, 0] — the firmware only emits fused orientation.

Standalone demo:
    python3 src/pen_receiver.py [port]

Press ctrl-c to quit. Press the pen's button once to start recording on the pen.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from typing import Optional

import serial
import serial.tools.list_ports

import config

FSR_MAX = 4095.0  # 12-bit ADC
# FSRs are exponentially sensitive — a light touch saturates the top of the
# range. sqrt() compresses the top and expands the bottom; a small deadband
# kills the resting noise floor so bars don't twitch when untouched.
FSR_DEADBAND = 0.015


def _fsr_curve(raw: int) -> float:
    norm = max(0.0, min(1.0, raw / FSR_MAX))
    if norm < FSR_DEADBAND:
        return 0.0
    return norm ** 0.5


_LINE_PATTERN = re.compile(
    r"roll\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"pitch\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"yaw\s*=\s*(-?\d+(?:\.\d+)?)"
    r"(.*)$"
)
_FSR_PATTERN = re.compile(r"fsr\s*(\d+)\s*=\s*(-?\d+)")


def find_port(vid: int = 9025) -> Optional[str]:
    """Return the first Arduino-like serial port, or None."""
    for p in serial.tools.list_ports.comports():
        if p.vid == vid:
            return p.device
    for p in serial.tools.list_ports.comports():
        if p.description and ("Nano" in p.description or "Arduino" in p.description):
            return p.device
    return None


def parse_line(line: str) -> Optional[dict]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m = _LINE_PATTERN.match(s)
    if not m:
        return None
    roll = float(m.group(1))
    pitch = float(m.group(2))
    yaw = float(m.group(3))
    fsrs = {int(n): int(v) for n, v in _FSR_PATTERN.findall(m.group(4))}
    if not fsrs:
        return None
    pressures = [_fsr_curve(fsrs.get(i, 0)) for i in range(4)]
    raw_list = [fsrs.get(i, 0) for i in range(max(fsrs) + 1)]
    return {
        "t": time.time(),
        "pressure": pressures,
        "fsr_raw": raw_list,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        # firmware emits fused orientation only — stuff it into the IMU slots
        # (ax/ay/az = roll/pitch/yaw; gx/gy/gz = 0) so the webapp shows live data.
        "imu": [roll, pitch, yaw, 0.0, 0.0, 0.0],
        "pen_down": max(pressures) > 0.05,
    }


class PenReceiver:
    def __init__(
        self,
        port: Optional[str] = None,
        baud: Optional[int] = None,
        buffer_size: int = 4000,
    ):
        self._port_arg = port
        self.baud = baud or getattr(config, "SERIAL_BAUD", 115200)
        self._buffer: deque[dict] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._ser: Optional[serial.Serial] = None
        self._buf: bytearray = bytearray()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.port: Optional[str] = None
        self.recording = False
        self.last_status: str = ""
        # EMA state for noise reduction
        self._ema_alpha = 0.35
        self._ema_p = [0.0, 0.0, 0.0, 0.0]
        self._ema_rpy = [0.0, 0.0, 0.0]
        self._ema_primed = False

    # -- lifecycle ---------------------------------------------------------

    def _resolve_port(self) -> str:
        if self._port_arg:
            return self._port_arg
        configured = getattr(config, "SERIAL_PORT", "") or None
        if configured:
            return configured
        found = find_port()
        if not found:
            raise RuntimeError("no Arduino-like serial port found; plug in the pen")
        return found

    def open(self) -> str:
        port = self._resolve_port()
        self._ser = serial.Serial(port, self.baud, timeout=0.1)
        self.connected = True
        self.port = port
        return port

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.connected = False

    # -- reading -----------------------------------------------------------

    def pump(self) -> int:
        """Read any available bytes and parse complete lines. Returns samples added."""
        if self._ser is None:
            return 0
        try:
            chunk = self._ser.read(2048)
        except Exception:
            return 0
        if not chunk:
            return 0
        self._buf.extend(chunk)
        added = 0
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl]).rstrip(b"\r").decode("utf-8", "replace")
            del self._buf[: nl + 1]
            if self._handle_line(line):
                added += 1
        return added

    def _handle_line(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("#"):
            self.last_status = s
            if "recording started" in s:
                self.recording = True
            elif "recording stopped" in s:
                self.recording = False
            return False
        sample = parse_line(s)
        if sample is None:
            return False
        self._smooth(sample)
        with self._lock:
            self._buffer.append(sample)
        return True

    def _smooth(self, sample: dict) -> None:
        """In-place EMA smoothing on pressure[4] and roll/pitch/yaw."""
        a = self._ema_alpha
        if not self._ema_primed:
            self._ema_p = list(sample["pressure"])
            self._ema_rpy = [sample["roll"], sample["pitch"], sample["yaw"]]
            self._ema_primed = True
        else:
            for i in range(4):
                self._ema_p[i] = a * sample["pressure"][i] + (1 - a) * self._ema_p[i]
            self._ema_rpy[0] = a * sample["roll"] + (1 - a) * self._ema_rpy[0]
            self._ema_rpy[1] = a * sample["pitch"] + (1 - a) * self._ema_rpy[1]
            self._ema_rpy[2] = a * sample["yaw"] + (1 - a) * self._ema_rpy[2]
        sample["pressure"] = list(self._ema_p)
        sample["roll"] = self._ema_rpy[0]
        sample["pitch"] = self._ema_rpy[1]
        sample["yaw"] = self._ema_rpy[2]
        sample["imu"] = [sample["roll"], sample["pitch"], sample["yaw"], 0.0, 0.0, 0.0]

    def start_thread(self) -> None:
        """Launch an OS thread that pumps the serial port continuously."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._thread.start()

    def _bg_loop(self) -> None:
        while not self._stop.is_set():
            self.pump()

    # -- readers -----------------------------------------------------------

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._buffer)


def _demo(port: Optional[str] = None) -> None:
    rx = PenReceiver(port=port)
    actual = rx.open()
    print(f"opened {actual} @ {rx.baud}")
    print("press the pen's button on the pen to start recording.")
    print("ctrl-c to quit.\n")
    rx.start_thread()
    try:
        last = 0.0
        last_status = ""
        while True:
            if rx.last_status and rx.last_status != last_status:
                last_status = rx.last_status
                print(f"[pen] {last_status}")
            s = rx.latest()
            if s and s["t"] - last > 0.1:
                last = s["t"]
                p = s["pressure"]
                print(
                    f"p=[{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {p[3]:.2f}] "
                    f"rpy=[{s['roll']:+7.2f} {s['pitch']:+7.2f} {s['yaw']:+7.2f}] "
                    f"down={s['pen_down']}"
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        print("\nclosed.")


if __name__ == "__main__":
    _demo(sys.argv[1] if len(sys.argv) > 1 else None)
