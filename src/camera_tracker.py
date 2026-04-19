"""Webcam pen-tip tracker via ink-appearance differencing.

Approach
--------
We don't track the pen body; we track NEW INK. Between consecutive frames,
pixels inside the paper rectangle that got darker correspond to new ink (or
the pen body shifting). The centroid of the darker-pixel mask approximates
where ink is currently being laid down.

Pipeline per frame:
  1. Auto-detect the largest bright quadrilateral (the paper) once, cache a
     binary paper mask. Press 'r' to redetect.
  2. Grayscale + small blur.
  3. diff = max(prev - curr, 0) — pixels that got darker since last frame.
  4. Mask to the paper ROI, threshold, morphological open.
  5. If enough darker pixels: tip = moments centroid, pen_drawing = True.
  6. Else: fall back to |prev - curr| thresholding to catch pen hovering
     (any motion inside paper), pen_drawing = False.
  7. Exponential moving average on (x, y) to damp jitter.

Output dict: {t, x, y, x_px, y_px, pen_drawing} with x/y normalized 0..1 of
the raw frame so the overlay lines up with the streamed webcam feed.

Standalone demo: python3 src/camera_tracker.py [camera_index]
Keys:
  r  redetect paper
  c  clear path
  q  quit
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config


# how dark a pixel-delta counts as "darkened"
INK_DELTA = 18
# min darkened pixels to believe we found ink this frame
MIN_INK_PIXELS = 35
# motion-fallback threshold (used when no ink detected)
MOTION_DELTA = 22
MIN_MOTION_PIXELS = 80
# area heuristics for paper detection (fraction of frame)
MIN_PAPER_FRAC = 0.05
# smoothing
EMA_ALPHA = 0.45


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as [tl, tr, br, bl]."""
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class CameraTracker:
    def __init__(self, camera_index: Optional[int] = None):
        self.camera_index = (
            camera_index if camera_index is not None else config.CAMERA_INDEX
        )
        self.cap: Optional[cv2.VideoCapture] = None
        self.paper_corners: Optional[np.ndarray] = None
        self.paper_mask: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.ema_pos: Optional[Tuple[float, float]] = None

    # ---- lifecycle -------------------------------------------------------

    def open(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap = cap

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reset(self) -> None:
        self.paper_corners = None
        self.paper_mask = None
        self.prev_gray = None
        self.ema_pos = None

    # ---- paper detection -------------------------------------------------

    def detect_paper(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # Otsu auto-picks the paper-vs-background split
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # OPEN with a large kernel kills thin features (cutting mat text/scale
        # lines) so only large bright regions survive — the paper.
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8)
        )
        # CLOSE fills inward notches where the pen / hand occludes the paper.
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8)
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        frame_area = frame.shape[0] * frame.shape[1]
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < MIN_PAPER_FRAC * frame_area:
                break
            # convex hull removes inward notches from pen / hand occlusion
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            # try progressively looser epsilons to land exactly 4 corners
            for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    return _order_corners(approx.reshape(4, 2))
        return None

    def set_paper_corners(
        self, corners: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> None:
        h, w = frame_shape[:2]
        self.paper_corners = corners
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        self.paper_mask = mask
        self.prev_gray = None
        self.ema_pos = None

    def _ensure_paper(self, frame: np.ndarray) -> None:
        if self.paper_corners is not None:
            return
        corners = self.detect_paper(frame)
        if corners is not None:
            self.set_paper_corners(corners, frame.shape)

    # ---- core tracking ---------------------------------------------------

    def _masked(self, img: np.ndarray) -> np.ndarray:
        if self.paper_mask is None:
            return img
        return cv2.bitwise_and(img, img, mask=self.paper_mask)

    def _find_ink_tip(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        # pixels that got darker this frame = new ink (and moving pen body)
        diff_signed = prev_gray.astype(np.int16) - curr_gray.astype(np.int16)
        darker = np.clip(diff_signed, 0, 255).astype(np.uint8)
        darker = self._masked(darker)
        _, ink_mask = cv2.threshold(darker, INK_DELTA, 255, cv2.THRESH_BINARY)
        ink_mask = cv2.morphologyEx(
            ink_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        if int(np.count_nonzero(ink_mask)) < MIN_INK_PIXELS:
            return None, False
        m = cv2.moments(ink_mask)
        if m["m00"] <= 0:
            return None, False
        return (m["m10"] / m["m00"], m["m01"] / m["m00"]), True

    def _find_motion_tip(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        abs_diff = cv2.absdiff(prev_gray, curr_gray)
        abs_diff = self._masked(abs_diff)
        _, move_mask = cv2.threshold(
            abs_diff, MOTION_DELTA, 255, cv2.THRESH_BINARY
        )
        move_mask = cv2.morphologyEx(
            move_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        if int(np.count_nonzero(move_mask)) < MIN_MOTION_PIXELS:
            return None
        contours, _ = cv2.findContours(
            move_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        m = cv2.moments(largest)
        if m["m00"] <= 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    def read(self) -> Tuple[Optional[dict], Optional[np.ndarray]]:
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, None

        self._ensure_paper(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        tip_px: Optional[Tuple[float, float]] = None
        pen_drawing = False

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            tip_px, pen_drawing = self._find_ink_tip(self.prev_gray, gray)
            if tip_px is None:
                tip_px = self._find_motion_tip(self.prev_gray, gray)

        self.prev_gray = gray

        if tip_px is None:
            return None, frame

        # EMA smoothing
        if self.ema_pos is None:
            self.ema_pos = tip_px
        else:
            self.ema_pos = (
                EMA_ALPHA * tip_px[0] + (1 - EMA_ALPHA) * self.ema_pos[0],
                EMA_ALPHA * tip_px[1] + (1 - EMA_ALPHA) * self.ema_pos[1],
            )

        x, y = self.ema_pos
        h, w = frame.shape[:2]
        return (
            {
                "t": time.time(),
                "x": float(x / w),
                "y": float(y / h),
                "x_px": float(x),
                "y_px": float(y),
                "pen_drawing": bool(pen_drawing),
            },
            frame,
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


# --------------------------------------------------------------------------
# standalone demo
# --------------------------------------------------------------------------
def _demo(camera_index: Optional[int] = None) -> None:
    tracker = CameraTracker(camera_index=camera_index)
    tracker.open()
    print(
        f"camera {tracker.camera_index} opened. "
        "keys: r=redetect paper, c=clear path, q=quit"
    )

    # path history in raw-frame pixel coords
    path: list[tuple[int, int, bool]] = []

    fps_t = time.time()
    fps_n = 0
    fps = 0.0

    cv2.namedWindow("penDNA tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("penDNA tracker", 1280, 720)

    try:
        while True:
            point, frame = tracker.read()
            if frame is None:
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # paper outline
            if tracker.paper_corners is not None:
                pts = tracker.paper_corners.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts], True, (0, 255, 0), 3)

            # tip + path
            if point is not None:
                px = int(point["x"] * w)
                py = int(point["y"] * h)
                drawing = point["pen_drawing"]
                path.append((px, py, drawing))
                if len(path) > 4000:
                    path.pop(0)
                color = (0, 0, 255) if drawing else (180, 180, 180)
                cv2.circle(display, (px, py), 12, color, 2)
                print(
                    f"x={point['x']:.3f} y={point['y']:.3f} "
                    f"drawing={drawing}"
                )

            for i in range(1, len(path)):
                a, b = path[i - 1], path[i]
                if not (a[2] and b[2]):
                    continue
                cv2.line(display, a[:2], b[:2], (86, 110, 15), 2)

            # hud
            fps_n += 1
            if time.time() - fps_t >= 0.5:
                fps = fps_n / (time.time() - fps_t)
                fps_t = time.time()
                fps_n = 0
            lines = [
                f"fps:   {fps:4.1f}",
                f"paper: {'ok' if tracker.paper_corners is not None else 'not found'}",
                f"tip:   {'drawing' if (point and point['pen_drawing']) else ('hover' if point else 'none')}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (14, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, line, (14, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)

            cv2.imshow("penDNA tracker", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                path.clear()
                print("reset: paper will redetect on next frame")
            elif key == ord("c"):
                path.clear()
                print("path cleared")
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _demo(idx)
