"""Fuse timestamped sensor + path buffers into a single aligned stream."""
from __future__ import annotations

import numpy as np


SENSOR_KEYS = ("p1", "p2", "p3", "p4", "ax", "ay", "az", "gx", "gy", "gz", "pen_down")


def fuse(sensor_buffer, path_buffer):
    if not sensor_buffer or not path_buffer:
        return []

    path_t = np.asarray([p["t"] for p in path_buffer], dtype=float)
    path_x = np.asarray([p["x"] for p in path_buffer], dtype=float)
    path_y = np.asarray([p["y"] for p in path_buffer], dtype=float)

    order = np.argsort(path_t)
    path_t = path_t[order]
    path_x = path_x[order]
    path_y = path_y[order]

    sensor_t = np.asarray([s["t"] for s in sensor_buffer], dtype=float)

    idx_right = np.searchsorted(path_t, sensor_t, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(path_t) - 1)
    idx_right = np.clip(idx_right, 0, len(path_t) - 1)

    left_dt = np.abs(sensor_t - path_t[idx_left])
    right_dt = np.abs(path_t[idx_right] - sensor_t)
    pick_right = right_dt < left_dt
    nearest = np.where(pick_right, idx_right, idx_left)

    xs = path_x[nearest]
    ys = path_y[nearest]

    fused = []
    for i, s in enumerate(sensor_buffer):
        point = {"t": float(s["t"]), "x": float(xs[i]), "y": float(ys[i])}
        for k in SENSOR_KEYS:
            v = s.get(k, 0)
            point[k] = bool(v) if k == "pen_down" else float(v)
        fused.append(point)
    return fused


def _demo():
    rng = np.random.default_rng(0)
    n = 500
    ts = np.linspace(0.0, 5.0, n)

    sensor_buffer = []
    for i, t in enumerate(ts):
        sensor_buffer.append({
            "t": float(t + rng.normal(0, 0.0005)),
            "p1": 0.3 + 0.2 * np.sin(2 * t),
            "p2": 0.4 + 0.15 * np.cos(3 * t),
            "p3": 0.25 + 0.1 * np.sin(5 * t),
            "p4": 0.35 + 0.05 * np.cos(7 * t),
            "ax": rng.normal(0, 0.05) + 0.01 * np.sin(40 * t),
            "ay": rng.normal(0, 0.05) + 0.01 * np.cos(40 * t),
            "az": 9.81 + rng.normal(0, 0.02),
            "gx": rng.normal(0, 0.1),
            "gy": rng.normal(0, 0.1),
            "gz": rng.normal(0, 0.1),
            "pen_down": 1 if (i % 120) < 90 else 0,
        })

    path_ts = ts[::2] + rng.normal(0, 0.001, size=n // 2)
    path_buffer = []
    for t in path_ts:
        path_buffer.append({
            "t": float(t),
            "x": 100.0 + 50.0 * np.sin(t),
            "y": 100.0 + 50.0 * np.cos(t),
        })

    fused = fuse(sensor_buffer, path_buffer)
    print(f"fused {len(fused)} points from {len(sensor_buffer)} sensor + {len(path_buffer)} path")
    print("first 10:")
    for p in fused[:10]:
        print(f"  t={p['t']:.4f} x={p['x']:.2f} y={p['y']:.2f} "
              f"p1={p['p1']:.3f} ax={p['ax']:.4f} pen={p['pen_down']}")


if __name__ == "__main__":
    _demo()
