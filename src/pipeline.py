import logging
import time

log = logging.getLogger(__name__)


def _fuse(sensor_buffer, path_buffer):
    return {
        "sensor_count": len(sensor_buffer),
        "path_count": len(path_buffer),
        "sensors": list(sensor_buffer),
        "path": list(path_buffer),
    }


def _extract_style(fused):
    path = fused.get("path", [])
    sensors = fused.get("sensors", [])
    avg_pressure = 0.5
    if sensors:
        pressures = [sum(s.get("pressure", [0, 0, 0, 0])) / 4 for s in sensors]
        if pressures:
            avg_pressure = sum(pressures) / len(pressures)
    return {
        "avg_pressure": avg_pressure,
        "tilt": 0.0,
        "speed": 1.0,
        "jitter": 0.05,
        "sample_points": len(path),
    }


def _generate_skeleton(prompt):
    return [
        {"type": "path", "d": "M 100 100 L 200 200", "prompt": prompt},
        {"type": "path", "d": "M 200 100 C 250 150, 150 250, 200 300"},
    ]


def _physics_draw(skeleton, style):
    strokes = []
    for i, el in enumerate(skeleton):
        strokes.append({
            "id": i,
            "d": el.get("d", ""),
            "width": 2 + style.get("avg_pressure", 0.5) * 6,
            "opacity": min(1.0, 0.4 + style.get("avg_pressure", 0.5)),
            "jitter": style.get("jitter", 0.05),
        })
    return strokes


def run(prompt, sensor_buffer, path_buffer):
    t0 = time.time()
    fused = _fuse(sensor_buffer, path_buffer)
    style = _extract_style(fused)
    skeleton = _generate_skeleton(prompt)
    styled = _physics_draw(skeleton, style)
    log.info("pipeline done in %.3fs: %d strokes", time.time() - t0, len(styled))
    return {
        "prompt": prompt,
        "style": style,
        "strokes": styled,
    }
