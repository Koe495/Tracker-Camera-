from dataclasses import dataclass
from colorsys import rgb_to_hsv
from math import isfinite
from statistics import median
from typing import Protocol

from .frames import Frame, MAX_DEPTH_M, MIN_DEPTH_M, Vec3


@dataclass(frozen=True)
class Detection2D:
    # x0/y0 inclusive, x1/y1 exclusive in color pixels.
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    # Optional foreground samples as flat indices into the original color image.
    mask_pixels: tuple[int, ...] | None = None


class Detector(Protocol):
    def detect(self, bgrx: bytes, width: int, height: int) -> list[Detection2D]: ...


@dataclass(frozen=True)
class Measurement:
    detection: Detection2D
    xyz: Vec3
    valid_points: int
    depth_mad_m: float


@dataclass(frozen=True)
class BallDetector:
    hue_min: float = 8
    hue_max: float = 42
    min_saturation: float = 0.55
    min_value: int = 105
    stride: int = 2
    min_samples: int = 18

    def detect(self, bgrx: bytes, width: int, height: int) -> list[Detection2D]:
        stride = self.stride
        cols = (width + stride - 1) // stride
        rows = (height + stride - 1) // stride
        mask = bytearray(cols * rows)
        for gy, y in enumerate(range(0, height, stride)):
            for gx, x in enumerate(range(0, width, stride)):
                offset = 4 * (y * width + x)
                b, g, r = bgrx[offset:offset + 3]
                high, low = max(r, g, b), min(r, g, b)
                if high < self.min_value or high - low < self.min_saturation * high:
                    continue
                hue = 360 * rgb_to_hsv(r, g, b)[0]
                if self.hue_min <= self.hue_max:
                    inside = self.hue_min <= hue <= self.hue_max
                else:
                    inside = hue >= self.hue_min or hue <= self.hue_max
                mask[gy * cols + gx] = inside

        detections = []
        for start, active in enumerate(mask):
            if not active:
                continue
            mask[start] = 0
            stack = [start]
            component = []
            min_x, min_y, max_x, max_y = cols, rows, 0, 0
            while stack:
                index = stack.pop()
                gy, gx = divmod(index, cols)
                component.append((gy * width + gx) * stride)
                min_x, max_x = min(min_x, gx), max(max_x, gx)
                min_y, max_y = min(min_y, gy), max(max_y, gy)
                for nx, ny in ((gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)):
                    neighbor = ny * cols + nx
                    if 0 <= nx < cols and 0 <= ny < rows and mask[neighbor]:
                        mask[neighbor] = 0
                        stack.append(neighbor)
            w, h = max_x - min_x + 1, max_y - min_y + 1
            fill = len(component) / (w * h)
            aspect = min(w, h) / max(w, h)
            if len(component) < self.min_samples or fill < 0.45 or aspect < 0.55:
                continue
            detections.append(Detection2D(
                (min_x * stride, min_y * stride, min(width, (max_x + 1) * stride),
                 min(height, (max_y + 1) * stride)),
                fill * aspect, "ball", tuple(component),
            ))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


def localize(frame: Frame, detection: Detection2D, *, min_points: int = 8) -> Measurement | None:
    if detection.mask_pixels is None:
        x0, y0, x1, y1 = detection.bbox
        # For box-only plugins, sample the central half to reduce background.
        dx, dy = (x1 - x0) // 4, (y1 - y0) // 4
        pixels = (
            y * frame.width + x
            for y in range(max(0, y0 + dy), min(frame.height, y1 - dy), 2)
            for x in range(max(0, x0 + dx), min(frame.width, x1 - dx), 2)
        )
    else:
        pixels = iter(detection.mask_pixels)
    points = []
    for index in pixels:
        if not 0 <= index < frame.width * frame.height:
            continue
        x, y, z = frame.color_xyz_m[3 * index:3 * index + 3]
        if MIN_DEPTH_M <= z <= MAX_DEPTH_M and isfinite(x) and isfinite(y):
            points.append((x, y, z))
    if len(points) < min_points:
        return None
    depth = median(p[2] for p in points)
    mad = median(abs(p[2] - depth) for p in points)
    # Reject mixed foreground/background when there is no coherent depth mode.
    if mad > 0.06:
        return None
    inliers = [p for p in points if abs(p[2] - depth) <= max(0.035, 3 * mad)]
    if len(inliers) < min_points or 2 * len(inliers) < len(points):
        return None
    x, y, z = (median(axis) for axis in zip(*inliers))
    return Measurement(detection, (x, y, z), len(inliers), mad)
