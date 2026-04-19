import eventlet
eventlet.monkey_patch()

import base64
import logging
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from eventlet import tpool
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import pipeline
from src.camera_tracker import CameraTracker
from src.pen_receiver import PenReceiver

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
    "pen_thread": None,
    "camera_thread": None,
    "style_thread": None,
    "sensor_buffer": deque(maxlen=20000),
    "path_buffer": deque(maxlen=20000),
    "last_path": None,
    "t0": time.time(),
}

_pen = PenReceiver()


CAMERA_STREAM_WIDTH = 640      # resize before jpeg encode
CAMERA_FRAME_FPS = 15
CAMERA_POINT_FPS = 30
LIFT_GAP_SEC = 0.25            # missing detection -> emit lift


def _velocity_mm_s(prev, curr):
    if prev is None:
        return 0.0
    dt = max(1e-3, curr["t"] - prev["t"])
    # treat canvas as ~200mm wide for a believable mm/s readout
    dx = (curr["x"] - prev["x"]) * 200.0
    dy = (curr["y"] - prev["y"]) * 200.0
    return math.sqrt(dx * dx + dy * dy) / dt


def _pen_loop():
    log.info("pen serial thread starting")
    try:
        port = tpool.execute(_pen.open)
    except Exception as e:
        log.warning("pen open failed: %s", e)
        socketio.emit("pen_status", {"ok": False, "error": str(e)})
        return
    log.info("pen connected on %s", port)
    socketio.emit("pen_status", {"ok": True, "port": port})

    emit_interval = 1.0 / config.CAPTURE_RATE
    next_emit = time.time()
    last_sample_t = 0.0
    last_status_sent = ""

    while True:
        # reads + parses in a real OS thread via tpool so eventlet stays free
        tpool.execute(_pen.pump)

        now = time.time()
        if _pen.last_status and _pen.last_status != last_status_sent:
            last_status_sent = _pen.last_status
            socketio.emit("pen_status", {
                "ok": True, "port": _pen.port, "message": last_status_sent,
                "recording": _pen.recording,
            })

        if now >= next_emit:
            next_emit = now + emit_interval
            sample = _pen.latest()
            if sample is not None and sample["t"] > last_sample_t:
                last_sample_t = sample["t"]
                if state["capturing"]:
                    state["sensor_buffer"].append(sample)
                imu_list = sample["imu"]
                velocity = (
                    _velocity_mm_s(state.get("_prev_path"), state["last_path"])
                    if state["last_path"]
                    else 0.0
                )
                socketio.emit("sensor_data", {
                    "t": sample["t"],
                    "pressure": sample["pressure"],
                    "imu": {
                        "ax": imu_list[0], "ay": imu_list[1], "az": imu_list[2],
                        "gx": imu_list[3], "gy": imu_list[4], "gz": imu_list[5],
                    },
                    "pen_down": sample["pen_down"],
                    "velocity": velocity,
                    "latency": (now - sample["t"]) * 1000.0,
                    "roll": sample["roll"],
                    "pitch": sample["pitch"],
                    "yaw": sample["yaw"],
                })

        eventlet.sleep(0)


def _encode_frame(frame):
    h, w = frame.shape[:2]
    if w > CAMERA_STREAM_WIDTH:
        scale = CAMERA_STREAM_WIDTH / w
        frame = cv2.resize(frame, (CAMERA_STREAM_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _camera_loop():
    log.info("camera thread starting on index %d", config.CAMERA_INDEX)
    tracker = CameraTracker()
    try:
        tracker.open()
    except Exception as e:
        log.warning("camera open failed: %s", e)
        socketio.emit("camera_status", {"ok": False, "error": str(e)})
        return
    socketio.emit("camera_status", {"ok": True})

    frame_interval = 1.0 / CAMERA_FRAME_FPS
    last_frame_emit = 0.0
    last_point_t = 0.0
    lift_emitted = True

    while True:
        point, frame = tpool.execute(tracker.read)
        now = time.time()

        if frame is not None and now - last_frame_emit >= frame_interval:
            b64 = _encode_frame(frame)
            if b64 is not None:
                socketio.emit("camera_frame", b64)
            last_frame_emit = now

        if point is not None:
            last_point_t = now
            drawing = bool(point.get("pen_drawing", True))
            lift = not drawing
            if not lift:
                lift_emitted = False
            payload = {
                "x": point["x"],
                "y": point["y"],
                "t": now * 1000.0,
                "lift": lift,
            }
            socketio.emit("path_point", payload)
            if state["capturing"] and drawing:
                state["path_buffer"].append(point)
                state["_prev_path"] = state["last_path"]
                state["last_path"] = point
        elif not lift_emitted and now - last_point_t >= LIFT_GAP_SEC:
            socketio.emit("path_point", {"x": 0, "y": 0, "t": now * 1000.0, "lift": True})
            lift_emitted = True

        eventlet.sleep(1.0 / CAMERA_POINT_FPS)


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


def _ensure_camera_thread():
    if state["camera_thread"] is None:
        state["camera_thread"] = socketio.start_background_task(_camera_loop)


def _ensure_pen_thread():
    if state["pen_thread"] is None:
        state["pen_thread"] = socketio.start_background_task(_pen_loop)


def _ensure_threads():
    _ensure_pen_thread()
    _ensure_camera_thread()
    if state["style_thread"] is None:
        state["style_thread"] = socketio.start_background_task(_style_loop)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    latest = _pen.latest()
    return {
        "ok": True,
        "capturing": state["capturing"],
        "sensor_samples": len(state["sensor_buffer"]),
        "path_points": len(state["path_buffer"]),
        "pen": {
            "connected": _pen.connected,
            "port": _pen.port,
            "recording": _pen.recording,
            "buffered": len(_pen.snapshot()),
            "last_status": _pen.last_status,
            "latest": latest,
        },
    }


@socketio.on("connect")
def on_connect():
    log.info("client connected")
    _ensure_camera_thread()
    _ensure_pen_thread()
    socketio.emit("status", {"capturing": state["capturing"]})
    if _pen.connected:
        socketio.emit("pen_status", {
            "ok": True, "port": _pen.port, "recording": _pen.recording,
        })


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
            "color": "#1A1A1A",
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
