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
    "style_thread": None,
    "sensor_buffer": deque(maxlen=20000),
    "path_buffer": deque(maxlen=20000),
    "last_path": None,
    "t0": time.time(),
}


def _fake_ble_reader():
    t = 0.0
    while True:
        t += 1.0 / config.CAPTURE_RATE
        base = 0.5 + 0.2 * math.sin(t * 1.3)
        pressure = [
            max(0.05, min(0.95, base + random.uniform(-0.05, 0.05) + 0.1 * math.sin(t * 2 + i)))
            for i in range(4)
        ]
        imu = [
            0.4 * math.sin(t * 1.1) + random.uniform(-0.05, 0.05) + 0.3 * math.sin(t * 9.0),
            0.4 * math.cos(t * 0.9) + random.uniform(-0.05, 0.05) + 0.3 * math.cos(t * 9.0),
            9.8 + random.uniform(-0.05, 0.05),
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
        ]
        pen_down = (t % 4.0) < 3.0
        yield {"t": time.time(), "pressure": pressure, "imu": imu, "pen_down": pen_down}


def _fake_camera_tracker():
    # emit normalized coords in 0..1 for the frontend path overlay
    t = 0.0
    while True:
        t += 1.0 / config.CAPTURE_RATE
        x = 0.5 + 0.28 * math.cos(t * 0.7) + random.uniform(-0.005, 0.005)
        y = 0.5 + 0.28 * math.sin(t * 0.7) + random.uniform(-0.005, 0.005)
        yield {"t": time.time(), "x": x, "y": y, "lift": False}


def _velocity_mm_s(prev, curr):
    if prev is None:
        return 0.0
    dt = max(1e-3, curr["t"] - prev["t"])
    # treat canvas as ~200mm wide for a believable mm/s readout
    dx = (curr["x"] - prev["x"]) * 200.0
    dy = (curr["y"] - prev["y"]) * 200.0
    return math.sqrt(dx * dx + dy * dy) / dt


def _ble_loop():
    log.info("ble thread started (mock)")
    gen = _fake_ble_reader()
    interval = 1.0 / config.CAPTURE_RATE
    while True:
        if state["capturing"]:
            sample = next(gen)
            state["sensor_buffer"].append(sample)
            imu_list = sample["imu"]
            velocity = _velocity_mm_s(state.get("_prev_path"), state["last_path"]) if state["last_path"] else 0.0
            socketio.emit("sensor_data", {
                "t": sample["t"],
                "pressure": sample["pressure"],
                "imu": {
                    "ax": imu_list[0], "ay": imu_list[1], "az": imu_list[2],
                    "gx": imu_list[3], "gy": imu_list[4], "gz": imu_list[5],
                },
                "pen_down": sample["pen_down"],
                "velocity": velocity,
                "latency": 8 + random.uniform(-2, 3),
            })
        eventlet.sleep(interval)


def _camera_loop():
    log.info("camera thread started (mock)")
    gen = _fake_camera_tracker()
    interval = 1.0 / config.CAPTURE_RATE
    while True:
        if state["capturing"]:
            point = next(gen)
            state["path_buffer"].append(point)
            state["_prev_path"] = state["last_path"]
            state["last_path"] = point
            socketio.emit("path_point", {
                "x": point["x"], "y": point["y"],
                "t": point["t"] * 1000.0,
                "lift": point.get("lift", False),
            })
        eventlet.sleep(interval)


def _style_loop():
    """Emit a live style_update (grip fingerprint + tremor spectrum) while capturing."""
    log.info("style thread started")
    while True:
        if state["capturing"] and len(state["sensor_buffer"]) >= 30:
            recent = list(state["sensor_buffer"])[-256:]
            grip = [0.0, 0.0, 0.0, 0.0]
            for s in recent:
                for i in range(4):
                    grip[i] += s["pressure"][i]
            grip = [g / len(recent) for g in grip]

            # lightweight tremor spectrum from imu[0] (ax): naive dft over 40 bins
            ax = [s["imu"][0] for s in recent]
            n = len(ax)
            bins = 40
            spec = []
            mean = sum(ax) / n
            centered = [v - mean for v in ax]
            for k in range(1, bins + 1):
                re = im = 0.0
                for i, v in enumerate(centered):
                    ang = -2.0 * math.pi * k * i / n
                    re += v * math.cos(ang)
                    im += v * math.sin(ang)
                spec.append(math.sqrt(re * re + im * im) / n)
            peak = max(spec) or 1.0
            spec = [min(1.0, s / peak) for s in spec]

            socketio.emit("style_update", {
                "grip": grip,
                "tremor": spec,
                "user": "USER_A",
                "profiles": 1,
            })
        eventlet.sleep(1.0)


def _ensure_threads():
    if state["ble_thread"] is None:
        state["ble_thread"] = socketio.start_background_task(_ble_loop)
    if state["camera_thread"] is None:
        state["camera_thread"] = socketio.start_background_task(_camera_loop)
    if state["style_thread"] is None:
        state["style_thread"] = socketio.start_background_task(_style_loop)


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
    state["last_path"] = None
    state["_prev_path"] = None
    log.info("buffers cleared")
    socketio.emit("cleared", {})


def _emit_generation(result, canvas):
    """Normalize physics-pen strokes to 0..1 canvas space and stream them."""
    strokes = result.get("strokes", [])
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        socketio.emit("generation_done", {})
        return

    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(1e-6, maxx - minx)
    h = max(1e-6, maxy - miny)
    scale = 0.8 / max(w, h)
    ox = 0.5 - (minx + w / 2) * scale
    oy = 0.5 - (miny + h / 2) * scale

    style = result.get("style", {})
    jitter = min(2.0, float(style.get("jerkiness", 0.2)) * 0.8)
    thickness = 1.6 + 1.2 * float(style.get("grip_asymmetry", 0.1))

    for stroke in strokes:
        pts = [{
            "x": p["x"] * scale + ox,
            "y": p["y"] * scale + oy,
            "pressure": float(p.get("pressure", 0.5)),
            "t": float(p.get("t", 0.0)),
        } for p in stroke]
        socketio.emit("stroke_data", {
            "points": pts,
            "color": "#E8E6E1",
            "thickness": thickness,
            "jitter": jitter,
            "speed_ms": 500,
            "canvas": canvas,
        })
        eventlet.sleep(0.05)
    socketio.emit("generation_done", {})


@socketio.on("generate")
def on_generate(data):
    prompt = (data or {}).get("prompt", "")
    split = bool((data or {}).get("split", False))
    canvas = (data or {}).get("canvas") or ("B" if split else "A")
    log.info("generate: %r (canvas=%s)", prompt, canvas)
    socketio.emit("generation_start", {"canvas": canvas})
    try:
        result = pipeline.run(prompt, state["sensor_buffer"], state["path_buffer"])
        _emit_generation(result, canvas)
    except Exception as e:
        log.exception("pipeline failed")
        socketio.emit("generation_error", {"error": str(e)})
        socketio.emit("generation_done", {})


def main():
    log.info("starting penDNA server on %s:%d", config.HOST, config.PORT)
    socketio.run(app, host=config.HOST, port=config.PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
