"""Metric ROI measurements and conservative depth-surface alerts.

No semantic detector is assumed. Alerts describe observed nearby surfaces,
not a free-space guarantee or identification of a chair/person.
"""
from collections import deque
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Measurement:
    bbox: tuple
    status: str
    median_mm: float | None
    std_mm: float | None
    valid_ratio: float
    valid_points: int


def measure_roi(depth, bbox, min_mm=800, max_mm=4000, min_ratio=0.5):
    """Bounding box is x,y,width,height in the ALIGNED RGB pixel grid."""
    h, w = depth.shape
    x, y, bw, bh = map(int, bbox)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if bw <= 0 or bh <= 0 or x0 >= x1 or y0 >= y1:
        return Measurement((x0, y0, 0, 0), "INVALID_ROI", None, None, 0.0, 0)
    region = depth[y0:y1, x0:x1]
    good = region[(region >= min_mm) & (region <= max_mm) & (region > 0)]
    ratio = float(good.size / region.size)
    enough = good.size >= 9 and ratio >= min_ratio
    return Measurement((x0, y0, x1 - x0, y1 - y0), "OK" if enough else "INSUFFICIENT_DEPTH",
                       float(np.median(good)) if enough else None,
                       float(np.std(good)) if enough else None, ratio, int(good.size))


@dataclass
class RGBDPacket:
    """Single capture owner passes this packet to OCR/detection adapters."""
    frame_id: int
    color_timestamp_ms: int
    depth_timestamp_ms: int
    color_bgr: np.ndarray
    depth_aligned_mm: np.ndarray
    source: str

    def measure_detection(self, detection_frame_id, bbox):
        if detection_frame_id != self.frame_id:
            raise ValueError("Detection belongs to a different frame; stale depth must not be used.")
        return measure_roi(self.depth_aligned_mm, bbox)


def fit_floor(xyz, seed=7):
    """Fit a floor only on request, with a roughly level fixed camera.

    Kinect camera frame: Y up, Z forward. Require substantial bottom-half
    support and a plane 0.25..2.5m below the camera; otherwise report unknown.
    """
    h = xyz.shape[0]
    points = xyz[h // 2::3, ::3].reshape(-1, 3)
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] >= 0.4)
             & (points[:, 2] <= 4) & (points[:, 1] < -0.1))
    points = points[valid]
    if len(points) < 200:
        return None
    rng = np.random.default_rng(seed)
    if len(points) > 3000:
        points = points[rng.choice(len(points), 3000, replace=False)]
    best, score = None, 0
    for _ in range(100):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-6:
            continue
        n = n / length
        if n[1] < 0:
            n = -n
        d = -float(n @ a)
        if n[1] < 0.86 or not 0.25 < d < 2.5:
            continue
        inliers = np.abs(points @ n + d) < 0.035
        if int(inliers.sum()) > score:
            best, score = inliers, int(inliers.sum())
    if best is None or score < max(200, int(0.35 * len(points))):
        return None
    selected = points[best]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    offset = -float(normal @ center)
    if normal[1] < 0.86 or not 0.25 < offset < 2.5:
        return None
    return np.r_[normal, offset].astype(np.float32)


@dataclass
class Zone:
    direction: str
    status: str
    distance_mm: float | None
    coverage: float


class ObstacleMonitor:
    def __init__(self, warning_mm=1500, danger_mm=1000, confirm_frames=3):
        self.warning_mm, self.danger_mm = warning_mm, danger_mm
        self.confirm_frames = confirm_frames
        self.floor = None
        self.histories = {name: deque(maxlen=confirm_frames) for name in ("LEFT", "CENTER", "RIGHT")}

    def calibrate_floor(self, xyz):
        self.floor = fit_floor(xyz)
        for history in self.histories.values():
            history.clear()
        return self.floor is not None

    def update(self, native_depth, xyz, min_mm=800, max_mm=4000):
        h, w = native_depth.shape
        valid = ((native_depth >= min_mm) & (native_depth <= max_mm)
                 & np.isfinite(xyz).all(axis=2))
        floor_mask = np.zeros((h, w), bool)
        if self.floor is not None:
            residual = np.abs(xyz @ self.floor[:3] + self.floor[3])
            floor_mask = valid & (residual < 0.06)
        results = []
        # Limited observation band. Does not cover drop-offs or the full walking path.
        top, bottom = int(h * .15), int(h * .85)
        for i, name in enumerate(self.histories):
            start, end = i * w // 3, (i + 1) * w // 3
            region = native_depth[top:bottom, start:end]
            good = valid[top:bottom, start:end]
            coverage = float(good.mean())
            near = good & ~floor_mask[top:bottom, start:end] & (region <= self.warning_mm)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), 8)
            distances = []
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] >= max(30, int(region.size * .002)):
                    distances.append(float(np.percentile(region[labels == label], 10)))
            distance = min(distances) if distances else None
            raw = ("UNKNOWN" if coverage < .30 else
                   "DANGER" if distance is not None and distance <= self.danger_mm else
                   "NEAR" if distance is not None else "NO_NEAR_SURFACE")
            history = self.histories[name]
            if raw == "UNKNOWN":
                history.clear()
                results.append(Zone(name, raw, None, coverage))
                continue
            history.append(raw)
            stable = len(history) == self.confirm_frames and len(set(history)) == 1
            # Never show a previous clear state while an alert is still being confirmed.
            status = raw if stable else "CHECKING"
            results.append(Zone(name, status, distance, coverage))
        return results


class Evaluation:
    """Append individual samples plus a summary; never invent ground truth."""
    def __init__(self, output_dir, reference_mm, sample_count, source):
        self.output_dir = Path(output_dir)
        self.reference_mm = reference_mm
        self.sample_count = sample_count
        self.source = source
        self.rows = []

    def add(self, packet, measurement):
        row = dict(frame_id=packet.frame_id, color_timestamp_ms=packet.color_timestamp_ms,
                   depth_timestamp_ms=packet.depth_timestamp_ms, source=self.source,
                   reference_mm=self.reference_mm, **asdict(measurement))
        row["error_mm"] = (measurement.median_mm - self.reference_mm
                           if measurement.median_mm is not None else None)
        self.rows.append(row)
        return len(self.rows) >= self.sample_count

    def save(self, name):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / name
        with target.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        values = [row["median_mm"] for row in self.rows if row["median_mm"] is not None]
        errors = np.array(values) - self.reference_mm
        summary = dict(source=self.source, reference_mm=self.reference_mm,
            total_samples=len(self.rows), valid_samples=len(values),
            mean_error_mm=float(errors.mean()) if len(values) else None,
            mae_mm=float(np.abs(errors).mean()) if len(values) else None,
            temporal_std_mm=float(np.std(values)) if values else None,
            mean_valid_ratio=float(np.mean([r["valid_ratio"] for r in self.rows])))
        target.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return target.with_suffix(".csv")
