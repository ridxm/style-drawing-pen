import eventlet
eventlet.monkey_patch()

import logging
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

from flask import Flask, send_from_directory
from flask_socketio import SocketIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("penDNA.server")

app = Flask(
    __name__,
    template_folder=str(ROOT / "src" / "templates"),
    static_folder=str(ROOT / "src" / "static"),
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

state = {
    "capturing": False,
    "ble_thread": None,
    "camera_thread": None,
    "sensor_buffer": deque(maxlen=20000),
    "path_buffer": deque(maxlen=20000),
    "t0": time.time(),
}


def _fake_ble_reader():
    t = 0.0
    while True:
        t += 1.0 / config.CAPTURE_RATE
        base = 0.5 + 0.2 * math.sin(t * 1.3)
        pressure = [
            max(0.2, min(0.8, base + random.uniform(-0.05, 0.05) + 0.1 * math.sin(t * 2 + i)))
            for i in range(4)
        ]
        imu = [
            0.02 * math.sin(t * 1.1) + random.uniform(-0.01, 0.01),
            0.02 * math.cos(t * 0.9) + random.uniform(-0.01, 0.01),
            9.8 + random.uniform(-0.02, 0.02),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ]
        yield {"t": time.time(), "pressure": pressure, "imu": imu}


def _fake_camera_tracker():
    t = 0.0
    cx, cy = 400.0, 300.0
    r = 150.0
    while True:
        t += 1.0 / config.CAPTURE_RATE
        x = cx + r * math.cos(t * 0.7) + random.uniform(-1.5, 1.5)
        y = cy + r * math.sin(t * 0.7) + random.uniform(-1.5, 1.5)
        yield {"t": time.time(), "x": x, "y": y}


def _ble_loop():
    log.info("ble thread started (mock)")
    gen = _fake_ble_reader()
    interval = 1.0 / config.CAPTURE_RATE
    while True:
        if state["capturing"]:
            sample = next(gen)
            state["sensor_buffer"].append(sample)
            socketio.emit("sensor_data", sample)
        eventlet.sleep(interval)


def _camera_loop():
    log.info("camera thread started (mock)")
    gen = _fake_camera_tracker()
    interval = 1.0 / config.CAPTURE_RATE
    while True:
        if state["capturing"]:
            point = next(gen)
            state["path_buffer"].append(point)
            socketio.emit("path_point", point)
        eventlet.sleep(interval)


def _ensure_threads():
    if state["ble_thread"] is None:
        state["ble_thread"] = socketio.start_background_task(_ble_loop)
    if state["camera_thread"] is None:
        state["camera_thread"] = socketio.start_background_task(_camera_loop)


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/health")
def health():
    return {
        "ok": True,
        "capturing": state["capturing"],
        "sensor_samples": len(state["sensor_buffer"]),
        "path_points": len(state["path_buffer"]),
    }


@socketio.on("connect")
def on_connect():
    log.info("client connected")
    socketio.emit("status", {"capturing": state["capturing"]})


@socketio.on("disconnect")
def on_disconnect():
    log.info("client disconnected")


@socketio.on("start_capture")
def on_start():
    state["capturing"] = True
    _ensure_threads()
    log.info("capture started")
    socketio.emit("status", {"capturing": True})


@socketio.on("stop_capture")
def on_stop():
    state["capturing"] = False
    log.info("capture stopped (buffers retained: %d sensor, %d path)",
             len(state["sensor_buffer"]), len(state["path_buffer"]))
    socketio.emit("status", {"capturing": False})


@socketio.on("clear")
def on_clear():
    state["sensor_buffer"].clear()
    state["path_buffer"].clear()
    log.info("buffers cleared")
    socketio.emit("cleared", {})


@socketio.on("generate")
def on_generate(data):
    prompt = (data or {}).get("prompt", "")
    log.info("generate: %r", prompt)
    try:
        result = pipeline.run(prompt, state["sensor_buffer"], state["path_buffer"])
        socketio.emit("generation_result", result)
    except Exception as e:
        log.exception("pipeline failed")
        socketio.emit("generation_error", {"error": str(e)})


def main():
    log.info("starting penDNA server on %s:%d", config.HOST, config.PORT)
    socketio.run(app, host=config.HOST, port=config.PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
