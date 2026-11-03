"""Frame pairing, depth rendering and a clearly labelled synthetic source."""
from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np

from kinect_v1 import Frame


class Pairer:
    """Pair each frame once; reject pairs beyond the device timestamp tolerance."""
    def __init__(self, tolerance_ms: int = 40):
        self.color = deque(maxlen=4)
        self.depth = deque(maxlen=4)
        self.tolerance_ms = tolerance_ms

    def add(self, color: Frame | None, depth: Frame | None):
        if color is not None:
            self.color.append(color)
        if depth is not None:
            self.depth.append(depth)
        while self.color and self.depth:
            delta = self.color[0].timestamp_ms - self.depth[0].timestamp_ms
            if abs(delta) <= self.tolerance_ms:
                return self.color.popleft(), self.depth.popleft()
            (self.color if delta < 0 else self.depth).popleft()
        return None


def depth_image(depth: np.ndarray, near_mm: int, far_mm: int, gray=False) -> np.ndarray:
    if not 0 < near_mm < far_mm:
        raise ValueError("Depth limits must satisfy 0 < near < far.")
    # Fixed metric range avoids frame-to-frame colour flicker from auto normalization.
    fraction = np.clip((depth.astype(np.float32) - near_mm) / (far_mm - near_mm), 0, 1)
    level = np.rint((1 - fraction) * 255).astype(np.uint8)
    result = (cv2.cvtColor(level, cv2.COLOR_GRAY2BGR) if gray
              else cv2.applyColorMap(level, cv2.COLORMAP_TURBO))
    result[(depth < near_mm) | (depth > far_mm)] = 0
    return result


def text(image, message, point, scale=0.55, color=(220, 228, 236)):
    cv2.putText(image, message, point, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def dashboard(color: Frame, depth: Frame, near_mm: int, far_mm: int,
              fps: float, demo: bool, gray: bool = False, status: str = "") -> np.ndarray:
    """Native 640x480 images stay unregistered; depth cursor uses depth coordinates."""
    canvas = np.full((646, 1308, 3), (24, 20, 17), np.uint8)
    text(canvas, "KINECT 1517  /  RGB + DEPTH", (18, 30), 0.8)
    text(canvas, "SIMULATION - NO CAMERA DATA" if demo else "LIVE SENSOR", (800, 29),
         0.65, (70, 190, 255) if demo else (120, 230, 130))
    text(canvas, "COLOR   |   640 x 480", (12, 62))
    delta = abs(color.timestamp_ms - depth.timestamp_ms)
    text(canvas, f"DEPTH   |   millimetres   |   pair delta {delta} ms", (662, 62))
    canvas[76:556, 8:648] = color.data
    colored = depth_image(depth.data, near_mm, far_mm, gray)
    cv2.drawMarker(colored, (320, 240), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    canvas[76:556, 660:1300] = colored
    center = int(depth.data[240, 320])
    valid = (depth.data >= near_mm) & (depth.data <= far_mm)
    distance = f"{center / 1000:.3f} m" if near_mm <= center <= far_mm else "N/A"
    text(canvas, f"{fps:4.1f} paired FPS", (14, 581))
    text(canvas, f"Depth center: {distance}   |   in range: {valid.mean():.1%}", (660, 581))
    # Legend follows exactly the same depth mapping as the depth image.
    values = np.linspace(near_mm, far_mm, 390).astype(np.uint16)[None, :]
    canvas[598:611, 790:1180] = np.repeat(depth_image(values, near_mm, far_mm, gray), 13, axis=0)
    text(canvas, f"{near_mm / 1000:.1f}m near", (660, 612), 0.45)
    text(canvas, f"{far_mm / 1000:.1f}m far", (1190, 612), 0.45)
    text(canvas, status or "Q / ESC: quit    G: color / gray depth    S: save RGB + raw depth", (14, 637), 0.48)
    return canvas


class DemoSource:
    """Synthetic RGB/depth moving objects for development; never a camera fallback."""
    def __init__(self):
        self.started = time.monotonic()
        self.last_tick = -1
        self.pending = None
        self.y, self.x = np.mgrid[:480, :640]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, stream):
        if stream == "depth":
            frame, self.pending = self.pending, None
            return frame
        elapsed = time.monotonic() - self.started
        tick = int(elapsed * 30)
        if tick == self.last_tick:
            return None
        self.last_tick = tick
        depth = (1200 + self.x * 3 + self.y).astype(np.uint16)
        color = np.zeros((480, 640, 3), np.uint8)
        color[:, :, 0] = (self.x / 3).astype(np.uint8)
        color[:, :, 1] = (self.y / 2).astype(np.uint8)
        color[:, :, 2] = 85
        cx = int(320 + 150 * np.sin(elapsed))
        mask = (self.x - cx) ** 2 + (self.y - 240) ** 2 < 85 ** 2
        depth[mask] = int(1000 + 120 * np.sin(elapsed * 2))
        color[mask] = (40, 175, 245)
        depth[30:70, 30:120] = 0  # invalid region, shown black
        text(color, "SYNTHETIC RGB", (20, 455), 0.9)
        self.pending = Frame(depth, tick * 33, tick)
        return Frame(color, tick * 33, tick)
