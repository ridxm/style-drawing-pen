"""Physics-based pen simulator driven by a style vector."""
from __future__ import annotations

import json
import math
import numpy as np


TOTAL_SAMPLES = 200
CORNER_CURV_QUANTILE = 0.85
TREMOR_SCALE = 40.0
JITTER_SCALE = 0.6
PRESSURE_MIN = 0.25
PRESSURE_MAX = 1.0
CORNER_SCALE = 1.0


def _catmull_rom_bezier(ctrl):
    ctrl = np.asarray(ctrl, dtype=float)
    if len(ctrl) == 1:
        return np.tile(ctrl, (TOTAL_SAMPLES, 1))
    padded = np.vstack([ctrl[:1], ctrl, ctrl[-1:]])
    n_seg = len(ctrl) - 1
    per_seg = max(2, TOTAL_SAMPLES // n_seg)
    pts = []
    for i in range(n_seg):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2
        ts = np.linspace(0.0, 1.0, per_seg, endpoint=(i == n_seg - 1))
        one = 1.0 - ts
        curve = (one ** 3)[:, None] * b0 \
            + (3 * one ** 2 * ts)[:, None] * b1 \
            + (3 * one * ts ** 2)[:, None] * b2 \
            + (ts ** 3)[:, None] * b3
        pts.append(curve)
    all_pts = np.vstack(pts)
    if len(all_pts) != TOTAL_SAMPLES:
        xp = np.linspace(0.0, 1.0, len(all_pts))
        xs = np.linspace(0.0, 1.0, TOTAL_SAMPLES)
        all_pts = np.column_stack([
            np.interp(xs, xp, all_pts[:, 0]),
            np.interp(xs, xp, all_pts[:, 1]),
        ])
    return all_pts


def _tangents(pts):
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    mag = np.sqrt(dx * dx + dy * dy) + 1e-9
    return np.column_stack([dx / mag, dy / mag])


def _curvature(pts):
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5 + 1e-9
    return num / den


def _velocity_profile(n, v_mean, v_std, accel_curve, rng):
    accel_curve = np.asarray(accel_curve, dtype=float)
    if accel_curve.size == 0 or (accel_curve.max() - accel_curve.min()) < 1e-9:
        accel_curve = np.linspace(0.0, 1.0, 20)

    ramp_len = max(3, int(0.15 * n))
    ramp_x = np.linspace(0.0, 1.0, ramp_len)
    accel_x = np.linspace(0.0, 1.0, len(accel_curve))
    ramp = np.interp(ramp_x, accel_x, accel_curve)
    ramp = (ramp - ramp.min()) / max(1e-9, ramp.max() - ramp.min())

    cruise_len = n - 2 * ramp_len
    cruise = np.ones(max(0, cruise_len))
    if cruise_len > 0 and v_std > 0:
        cruise = cruise + rng.normal(0, 0.1, size=cruise_len)

    shape = np.concatenate([ramp, cruise, ramp[::-1]])
    if len(shape) != n:
        shape = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(shape)), shape)

    v = v_mean * np.clip(shape, 0.05, None)
    return v


class PhysicsPen:
    def __init__(self, style_vector):
        self.style = style_vector
        self.rng = np.random.default_rng(42)

    def _draw_stroke(self, ctrl_pts, t_start):
        pts = _catmull_rom_bezier(ctrl_pts)
        tangent = _tangents(pts)
        perp = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        curv = _curvature(pts)

        v_mean = max(1e-3, float(self.style.get("velocity_mean", 100.0)))
        v_std = float(self.style.get("velocity_std", 0.0))
        accel_curve = self.style.get("acceleration_curve", [])
        speed = _velocity_profile(TOTAL_SAMPLES, v_mean, v_std, accel_curve, self.rng)

        seg_dx = np.diff(pts[:, 0], prepend=pts[0, 0])
        seg_dy = np.diff(pts[:, 1], prepend=pts[0, 1])
        seg_len = np.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        dt = seg_len / np.clip(speed, 1e-3, None)
        cum_t = np.cumsum(dt) + t_start

        f = float(self.style.get("tremor_peak_freq", 0.0))
        tp = float(self.style.get("tremor_power", 0.0))
        tremor_amp = tp * TREMOR_SCALE
        tremor_offset = tremor_amp * np.sin(2.0 * np.pi * f * (cum_t - t_start))

        jerk = float(self.style.get("jerkiness", 0.0))
        jitter_sigma = min(2.0, jerk * JITTER_SCALE)
        jitter = self.rng.normal(0.0, jitter_sigma, size=(TOTAL_SAMPLES, 2)) if jitter_sigma > 0 else np.zeros((TOTAL_SAMPLES, 2))

        displaced = pts + perp * tremor_offset[:, None] + jitter

        corner_mag = float(self.style.get("corner_behavior", 0.0)) * CORNER_SCALE
        if corner_mag > 0 and TOTAL_SAMPLES > 5:
            thr = np.quantile(curv, CORNER_CURV_QUANTILE)
            hi = curv > thr
            displaced[hi] += tangent[hi] * corner_mag

        pres_dyn = np.asarray(self.style.get("pressure_dynamics", []), dtype=float)
        if pres_dyn.size == 0:
            pressure = np.full(TOTAL_SAMPLES, (PRESSURE_MIN + PRESSURE_MAX) / 2.0)
        else:
            xp = np.linspace(0.0, 1.0, len(pres_dyn))
            xs = np.linspace(0.0, 1.0, TOTAL_SAMPLES)
            shape = np.interp(xs, xp, pres_dyn)
            shape = np.clip(shape, 0.0, 1.0)
            pressure = PRESSURE_MIN + shape * (PRESSURE_MAX - PRESSURE_MIN)

        out = []
        for i in range(TOTAL_SAMPLES):
            out.append({
                "x": float(displaced[i, 0]),
                "y": float(displaced[i, 1]),
                "pressure": float(pressure[i]),
                "speed": float(speed[i]),
                "t": float(cum_t[i]),
            })
        return out, float(cum_t[-1])

    def draw_from_skeleton(self, stroke_groups):
        rhythm = self.style.get("stroke_rhythm", {})
        mean_gap = float(rhythm.get("mean_gap", 0.2))
        std_gap = float(rhythm.get("std_gap", 0.05))

        out_strokes = []
        t_cursor = 0.0
        for i, ctrl in enumerate(stroke_groups):
            if i > 0:
                gap = max(0.0, self.rng.normal(mean_gap, max(0.0, std_gap)))
                t_cursor += gap
            stroke, t_end = self._draw_stroke(ctrl, t_cursor)
            out_strokes.append(stroke)
            t_cursor = t_end
        return out_strokes


def _moderate_style():
    return {
        "grip_fingerprint": [0.35, 0.42, 0.28, 0.38],
        "grip_asymmetry": 0.05,
        "pressure_dynamics": (0.5 + 0.45 * np.sin(np.linspace(0, 2 * np.pi, 50))).tolist(),
        "tremor_peak_freq": 8.0,
        "tremor_power": 0.08,
        "velocity_mean": 140.0,
        "velocity_std": 35.0,
        "velocity_percentiles": [60, 90, 140, 190, 230],
        "jerkiness": 0.8,
        "curvature_pressure_corr": 0.15,
        "stroke_rhythm": {
            "mean_stroke_duration": 0.9,
            "std_stroke_duration": 0.15,
            "mean_gap": 0.25,
            "std_gap": 0.06,
        },
        "direction_bias": [0.12] * 8,
        "corner_behavior": 1.8,
        "acceleration_curve": np.clip(
            1 - np.cos(np.linspace(0, np.pi / 2, 20)), 0, 1
        ).tolist(),
    }


def _house_skeleton():
    return [
        [(20.0, 10.0), (20.0, 80.0)],
        [(120.0, 10.0), (120.0, 80.0)],
        [(20.0, 10.0), (120.0, 10.0)],
        [(20.0, 80.0), (70.0, 130.0), (120.0, 80.0)],
    ]


def _demo():
    style = _moderate_style()
    pen = PhysicsPen(style)
    strokes = pen.draw_from_skeleton(_house_skeleton())

    out_path = "demo_output.json"
    with open(out_path, "w") as fh:
        json.dump({"strokes": strokes, "style": style}, fh, indent=2)
    total_pts = sum(len(s) for s in strokes)
    t_last = strokes[-1][-1]["t"]
    print(f"wrote {out_path}: {len(strokes)} strokes, {total_pts} points, duration {t_last:.2f}s")

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.collections import LineCollection  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 7))
        for stroke in strokes:
            xs = [p["x"] for p in stroke]
            ys = [p["y"] for p in stroke]
            prs = [p["pressure"] for p in stroke]
            segs = [[(xs[i], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)]
            widths = [3.0 * (prs[i] + prs[i + 1]) / 2.0 for i in range(len(xs) - 1)]
            lc = LineCollection(segs, linewidths=widths, colors="black")
            ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title("penDNA physics demo — moderate style")
        plt.savefig("demo_output.png", dpi=120, bbox_inches="tight")
        print("rendered demo_output.png")
    except ImportError:
        print("matplotlib not available — skipping render")


if __name__ == "__main__":
    _demo()
