"""End-to-end generation pipeline: fuse -> extract style -> skeleton -> physics render."""
from __future__ import annotations

import logging
import time

from src import data_fusion, style_extractor, physics_pen, svg_generator

log = logging.getLogger(__name__)


def _sensor_to_fused_shape(sensor):
    """server sensor_data uses pressure:[4] + imu:[6]; fuse() wants p1..p4/ax..gz."""
    p = sensor.get("pressure", [0.0, 0.0, 0.0, 0.0])
    imu = sensor.get("imu", [0.0, 0.0, 9.8, 0.0, 0.0, 0.0])
    p = list(p) + [0.0] * (4 - len(p))
    imu = list(imu) + [0.0] * (6 - len(imu))
    return {
        "t": sensor["t"],
        "p1": p[0], "p2": p[1], "p3": p[2], "p4": p[3],
        "ax": imu[0], "ay": imu[1], "az": imu[2],
        "gx": imu[3], "gy": imu[4], "gz": imu[5],
        "pen_down": sensor.get("pen_down", True),
    }


def run(prompt, sensor_buffer, path_buffer):
    t0 = time.time()
    sensors = [_sensor_to_fused_shape(s) for s in sensor_buffer]
    paths = list(path_buffer)
    fused = data_fusion.fuse(sensors, paths) if sensors and paths else []
    style = style_extractor.extract(fused)

    skeleton = svg_generator.generate(prompt)
    pen = physics_pen.PhysicsPen(style)
    strokes = pen.draw_from_skeleton(skeleton)

    log.info("pipeline: %d fused, %d skeleton, %d styled strokes in %.3fs",
             len(fused), len(skeleton), len(strokes), time.time() - t0)
    return {"prompt": prompt, "style": style, "strokes": strokes}
