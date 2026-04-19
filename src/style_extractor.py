"""Extract a style vector from fused pen data."""
from __future__ import annotations

import json
import numpy as np


def _to_arrays(fused):
    keys = ("t", "x", "y", "p1", "p2", "p3", "p4",
            "ax", "ay", "az", "gx", "gy", "gz")
    out = {k: np.asarray([f[k] for f in fused], dtype=float) for k in keys}
    out["pen_down"] = np.asarray([bool(f["pen_down"]) for f in fused])
    return out


def _segments(pen_down):
    idx = np.flatnonzero(np.diff(pen_down.astype(np.int8)) != 0) + 1
    edges = np.concatenate(([0], idx, [len(pen_down)]))
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        segs.append((a, b, bool(pen_down[a])))
    return segs


def _resample(y, n):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(n)
    if len(y) == 1:
        return np.full(n, y[0])
    xp = np.linspace(0.0, 1.0, len(y))
    xs = np.linspace(0.0, 1.0, n)
    return np.interp(xs, xp, y)


def _normalize01(v):
    v = np.asarray(v, dtype=float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def _peak_freq(signal, t, fmin=4.0, fmax=20.0):
    if len(signal) < 16 or len(t) != len(signal):
        return 0.0, 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0, 0.0
    fs = 1.0 / float(np.median(dt))
    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    if len(t_uni) < 16:
        return 0.0, 0.0
    sig_uni = np.interp(t_uni, t, signal)
    sig_uni = sig_uni - sig_uni.mean()
    win = np.hanning(len(sig_uni))
    spec = np.fft.rfft(sig_uni * win)
    freqs = np.fft.rfftfreq(len(sig_uni), d=1.0 / fs)
    mag = np.abs(spec)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0, 0.0
    idx = np.argmax(mag[band])
    peak_f = float(freqs[band][idx])
    peak_p = float(mag[band][idx] / len(sig_uni))
    return peak_f, peak_p


def _curvature(x, y):
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5 + 1e-9
    return num / den


def extract(fused):
    if not fused:
        return _empty_style()

    a = _to_arrays(fused)
    t, x, y = a["t"], a["x"], a["y"]
    pen = a["pen_down"]

    mean_p = [float(a["p1"].mean()), float(a["p2"].mean()),
              float(a["p3"].mean()), float(a["p4"].mean())]
    grip_asym = float(np.std(mean_p))

    total_p = a["p1"] + a["p2"] + a["p3"] + a["p4"]
    dt_all = np.gradient(t)
    dt_all[dt_all <= 0] = np.median(dt_all[dt_all > 0]) if (dt_all > 0).any() else 1.0
    pres_grad = np.gradient(total_p) / dt_all
    pres_dyn = _normalize01(_resample(pres_grad, 50)).tolist()

    accel_mag = np.sqrt(a["ax"] ** 2 + a["ay"] ** 2 + a["az"] ** 2)
    accel_ac = accel_mag - accel_mag.mean()
    peak_f, peak_p = _peak_freq(accel_ac, t, 4.0, 20.0)

    jerk = np.gradient(accel_mag) / dt_all
    jerkiness = float(np.sqrt(np.mean(jerk * jerk)))

    segs = _segments(pen)
    down_segs = [(a0, b0) for a0, b0, d in segs if d and b0 - a0 >= 3]
    up_segs = [(a0, b0) for a0, b0, d in segs if not d and b0 - a0 >= 1]

    all_vel = []
    all_ang = []
    stroke_durs = []
    gap_durs = []
    curv_accum = []
    pres_accum = []
    corner_overshoot = []
    accel_curves = []

    for a0, b0 in down_segs:
        xi, yi, ti = x[a0:b0], y[a0:b0], t[a0:b0]
        if len(xi) < 3:
            continue
        dxi = np.diff(xi)
        dyi = np.diff(yi)
        dti = np.diff(ti)
        dti[dti <= 0] = 1e-6
        vi = np.sqrt(dxi * dxi + dyi * dyi) / dti
        all_vel.append(vi)

        ang = np.arctan2(dyi, dxi)
        all_ang.append(ang)

        stroke_durs.append(float(ti[-1] - ti[0]))

        k = _curvature(xi, yi)
        avg_p = (a["p1"][a0:b0] + a["p2"][a0:b0]
                 + a["p3"][a0:b0] + a["p4"][a0:b0]) / 4.0
        curv_accum.append(k)
        pres_accum.append(avg_p)

        if len(k) > 4:
            thr = np.quantile(k, 0.9)
            hi = np.flatnonzero(k > thr)
            for j in hi:
                lo = max(0, j - 2)
                hi2 = min(len(xi) - 1, j + 2)
                if hi2 - lo < 2:
                    continue
                x0, y0 = xi[lo], yi[lo]
                x1, y1 = xi[hi2], yi[hi2]
                px, py = xi[j], yi[j]
                vx, vy = x1 - x0, y1 - y0
                norm = np.sqrt(vx * vx + vy * vy) + 1e-9
                d = abs((vy * px - vx * py + x1 * y0 - y1 * x0)) / norm
                corner_overshoot.append(float(d))

        if len(vi) >= 4:
            ac = np.gradient(vi)
            accel_curves.append(_resample(ac, 20))

    for a0, b0 in up_segs:
        if b0 - a0 >= 1:
            gap_durs.append(float(t[min(b0, len(t) - 1)] - t[a0]))

    if all_vel:
        vel = np.concatenate(all_vel)
        vel = vel[np.isfinite(vel)]
    else:
        vel = np.array([0.0])

    vel_mean = float(vel.mean())
    vel_std = float(vel.std())
    vel_pct = [float(np.percentile(vel, p)) for p in (10, 25, 50, 75, 90)]

    if curv_accum and pres_accum:
        k_cat = np.concatenate(curv_accum)
        p_cat = np.concatenate(pres_accum)
        curv_p_corr = _pearson(k_cat, p_cat)
    else:
        curv_p_corr = 0.0

    if all_ang:
        ang = np.concatenate(all_ang)
        hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi))
        total = hist.sum()
        direction_bias = (hist / total).tolist() if total > 0 else [0.125] * 8
    else:
        direction_bias = [0.125] * 8

    corner_beh = float(np.mean(corner_overshoot)) if corner_overshoot else 0.0

    if accel_curves:
        avg_ac = np.mean(np.stack(accel_curves, axis=0), axis=0)
        accel_curve = _normalize01(avg_ac).tolist()
    else:
        accel_curve = [0.0] * 20

    rhythm = {
        "mean_stroke_duration": float(np.mean(stroke_durs)) if stroke_durs else 0.0,
        "std_stroke_duration": float(np.std(stroke_durs)) if stroke_durs else 0.0,
        "mean_gap": float(np.mean(gap_durs)) if gap_durs else 0.0,
        "std_gap": float(np.std(gap_durs)) if gap_durs else 0.0,
    }

    return {
        "grip_fingerprint": mean_p,
        "grip_asymmetry": grip_asym,
        "pressure_dynamics": pres_dyn,
        "tremor_peak_freq": peak_f,
        "tremor_power": peak_p,
        "velocity_mean": vel_mean,
        "velocity_std": vel_std,
        "velocity_percentiles": vel_pct,
        "jerkiness": jerkiness,
        "curvature_pressure_corr": curv_p_corr,
        "stroke_rhythm": rhythm,
        "direction_bias": direction_bias,
        "corner_behavior": corner_beh,
        "acceleration_curve": accel_curve,
    }


def _empty_style():
    return {
        "grip_fingerprint": [0.0] * 4,
        "grip_asymmetry": 0.0,
        "pressure_dynamics": [0.0] * 50,
        "tremor_peak_freq": 0.0,
        "tremor_power": 0.0,
        "velocity_mean": 0.0,
        "velocity_std": 0.0,
        "velocity_percentiles": [0.0] * 5,
        "jerkiness": 0.0,
        "curvature_pressure_corr": 0.0,
        "stroke_rhythm": {"mean_stroke_duration": 0.0, "std_stroke_duration": 0.0,
                           "mean_gap": 0.0, "std_gap": 0.0},
        "direction_bias": [0.125] * 8,
        "corner_behavior": 0.0,
        "acceleration_curve": [0.0] * 20,
    }


def _fake_fused(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 6.0, n)
    fused = []
    for i, t in enumerate(ts):
        stroke = (i // 150) % 2 == 0
        x = 100.0 + 60.0 * np.sin(0.8 * t) + rng.normal(0, 0.4)
        y = 100.0 + 60.0 * np.cos(0.8 * t) + rng.normal(0, 0.4)
        tremor_x = 0.6 * np.sin(2 * np.pi * 9.0 * t)
        tremor_y = 0.6 * np.cos(2 * np.pi * 9.0 * t)
        fused.append({
            "t": float(t),
            "x": float(x),
            "y": float(y),
            "p1": 0.3 + 0.1 * np.sin(2 * t) + rng.normal(0, 0.01),
            "p2": 0.4 + 0.08 * np.cos(3 * t) + rng.normal(0, 0.01),
            "p3": 0.25 + 0.05 * np.sin(5 * t) + rng.normal(0, 0.01),
            "p4": 0.35 + 0.03 * np.cos(7 * t) + rng.normal(0, 0.01),
            "ax": tremor_x + rng.normal(0, 0.05),
            "ay": tremor_y + rng.normal(0, 0.05),
            "az": 9.81 + rng.normal(0, 0.03),
            "gx": rng.normal(0, 0.1),
            "gy": rng.normal(0, 0.1),
            "gz": rng.normal(0, 0.1),
            "pen_down": 1 if stroke else 0,
        })
    return fused


def _demo():
    fused = _fake_fused()
    style = extract(fused)
    print(json.dumps(style, indent=2))


if __name__ == "__main__":
    _demo()
