"""Project tracked 3D points into RGB pixels using samples from the Kinect SDK."""

from dataclasses import dataclass
from struct import Struct

import numpy as np

from .tracking import HORIZONS

HEADER = Struct("<4sI")
SAMPLE = np.dtype([("xyz", "<f4", (3,)), ("uv", "<i4", (2,))])


@dataclass(frozen=True)
class ColorProjection:
    matrix: np.ndarray
    calibration_p95_px: float

    @classmethod
    def fit(cls, xyz, uv):
        """Fit a 3×4 projection with the denominator's Z coefficient fixed at 1."""
        xyz = np.asarray(xyz, dtype=float)
        uv = np.asarray(uv, dtype=float)
        if len(xyz) < 12 or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
            return None
        center = uv.mean(axis=0)
        target = (uv - center) / 320
        points = np.column_stack((xyz, np.ones(len(xyz))))
        system = np.zeros((2 * len(xyz), 11))
        system[0::2, :4] = points
        system[1::2, 4:8] = points
        system[:, 8:] = (-target[:, :, None] * points[:, None, [0, 1, 3]]).reshape(-1, 3)
        coefficients, _, rank, _ = np.linalg.lstsq(system, (target * xyz[:, 2:3]).ravel(), rcond=None)
        if rank != 11:
            return None
        denom = np.array([coefficients[8], coefficients[9], 1, coefficients[10]])
        matrix = np.vstack((coefficients[:4] * 320 + center[0] * denom,
                            coefficients[4:8] * 320 + center[1] * denom, denom))
        projected = points @ matrix.T
        if np.any(projected[:, 2] <= 0):
            return None
        errors = np.linalg.norm(projected[:, :2] / projected[:, 2:3] - uv, axis=1)
        p95 = float(np.percentile(errors, 95))
        return cls(matrix, p95) if p95 <= 4 else None

    def project(self, point):
        u, v, depth = self.matrix @ (*point, 1.0)
        if depth <= .1 or not np.isfinite((u, v, depth)).all():
            return None
        return u / depth, v / depth, depth


def read_projection(stream):
    header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("No RGB projection header from the Kinect bridge.")
    magic, count = HEADER.unpack(header)
    if magic != b"PRJ1" or not 12 <= count <= 256:
        raise ValueError("Incompatible Kinect bridge. Rebuild it with kinect/build.ps1.")
    payload = stream.read(count * SAMPLE.itemsize)
    if len(payload) != count * SAMPLE.itemsize:
        raise ValueError("Truncated RGB projection samples.")
    samples = np.frombuffer(payload, dtype=SAMPLE)
    valid = ((samples["uv"] >= 0) & (samples["uv"] < (640, 480))).all(axis=1)
    return ColorProjection.fit(samples["xyz"][valid], samples["uv"][valid])


@dataclass(frozen=True)
class BoxForecast:
    horizon_s: float
    delta_z_m: float
    scale: float | None
    bounds: tuple[float, float, float, float] | None


def forecast_boxes(track, projection):
    """Predict projected box bounds and physical depth change from the current state."""
    # A fresh RGB box is our image anchor at the current time. For coasting,
    # keep the previous measurement's projection so the image position advances.
    anchor_xyz = track.xyz if track.observed else track.box_xyz
    anchor = None
    if projection is not None and track.box is not None and anchor_xyz is not None:
        anchor = projection.project(anchor_xyz)
    for horizon, forecast in zip(HORIZONS, track.forecasts):
        delta_z = forecast[2] - track.xyz[2]
        projected = projection.project(forecast) if anchor is not None else None
        if projected is None:
            yield BoxForecast(horizon, delta_z, None, None)
            continue
        x0, y0, x1, y1 = track.box
        u, v, depth = projected
        ratio = anchor[2] / depth
        cx = (x0 + x1) / 2 + u - anchor[0]
        cy = (y0 + y1) / 2 + v - anchor[1]
        rx, ry = ratio * (x1 - x0) / 2, ratio * (y1 - y0) / 2
        yield BoxForecast(horizon, delta_z, ratio, (cx - rx, cy - ry, cx + rx, cy + ry))
