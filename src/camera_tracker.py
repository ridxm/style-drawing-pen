"""Webcam pen-tip tracker using OpenCV HSV filtering."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config

DEFAULT_HSV_LOW = (0, 120, 120)
DEFAULT_HSV_HIGH = (10, 255, 255)
OUTPUT_RANGE = 500.0


class CameraTracker:
    def __init__(
        self,
        camera_index: Optional[int] = None,
        hsv_low: Tuple[int, int, int] = DEFAULT_HSV_LOW,
        hsv_high: Tuple[int, int, int] = DEFAULT_HSV_HIGH,
        min_area: float = 80.0,
    ):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.min_area = min_area
        self.cap: Optional[cv2.VideoCapture] = None
        self._background: Optional[np.ndarray] = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.camera_index}")
        self.cap = cap

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set_hsv_range(
        self, low: Tuple[int, int, int], high: Tuple[int, int, int]
    ) -> None:
        self.hsv_low = np.array(low, dtype=np.uint8)
        self.hsv_high = np.array(high, dtype=np.uint8)

    def capture_background(self, samples: int = 10) -> None:
        """Store a clean background frame for motion-based fallback tracking."""
        if self.cap is None:
            self.open()
        frames = []
        for _ in range(samples):
            ok, frame = self.cap.read()
            if ok:
                frames.append(frame.astype(np.float32))
            time.sleep(0.02)
        if frames:
            self._background = np.mean(frames, axis=0).astype(np.uint8)

    def _find_tip_hsv(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return cx, cy

    def _find_tip_bgsub(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        if self._background is None:
            return None
        diff = cv2.absdiff(frame, self._background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    def read(self) -> Tuple[Optional[dict], Optional[np.ndarray]]:
        """Return (tracked_point, raw_frame). point is None if nothing found."""
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, None
        tip = self._find_tip_hsv(frame)
        if tip is None:
            tip = self._find_tip_bgsub(frame)
        if tip is None:
            return None, frame
        h, w = frame.shape[:2]
        x_norm = (tip[0] / w) * OUTPUT_RANGE
        y_norm = (tip[1] / h) * OUTPUT_RANGE
        return {"t": time.time(), "x": float(x_norm), "y": float(y_norm)}, frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


def _demo() -> None:
    tracker = CameraTracker()
    tracker.open()
    print("press q to quit")
    try:
        while True:
            point, frame = tracker.read()
            if frame is None:
                continue
            if point is not None:
                h, w = frame.shape[:2]
                px = int(point["x"] / OUTPUT_RANGE * w)
                py = int(point["y"] / OUTPUT_RANGE * h)
                cv2.circle(frame, (px, py), 10, (0, 255, 0), 2)
                print(f"x={point['x']:.1f} y={point['y']:.1f}")
            cv2.imshow("penDNA tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
