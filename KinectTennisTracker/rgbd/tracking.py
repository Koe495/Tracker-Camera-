from dataclasses import dataclass
from math import dist, isfinite
from typing import Protocol

from .detection import Detector, Measurement, localize
from .frames import Frame, Vec3

HORIZONS = (0.1, 0.2, 0.3, 0.4, 0.5)


class MotionModel(Protocol):
    acceleration_m_s2: Vec3
    process_noise_m2_s4: float


@dataclass(frozen=True)
class ConstantVelocity:
    acceleration_m_s2: Vec3 = (0.0, 0.0, 0.0)
    process_noise_m2_s4: float = 12.0


@dataclass(frozen=True)
class Ballistic:
    acceleration_m_s2: Vec3 = (0.0, -9.81, 0.0)
    process_noise_m2_s4: float = 4.0


@dataclass
class KalmanAxis:
    position: float
    velocity: float = 0.0
    pp: float = 0.025**2
    pv: float = 0.0
    vv: float = 4.0

    def predict(self, dt: float, acceleration: float, noise: float) -> None:
        half_t2 = dt * dt / 2
        self.position += dt * self.velocity + half_t2 * acceleration
        self.velocity += dt * acceleration
        self.pp += 2 * dt * self.pv + dt * dt * self.vv + noise * half_t2**2
        self.pv += dt * self.vv + noise * half_t2 * dt
        self.vv += noise * dt**2

    def update(self, measured: float, variance: float) -> None:
        residual = measured - self.position
        innovation = self.pp + variance
        kp, kv = self.pp / innovation, self.pv / innovation
        self.position += kp * residual
        self.velocity += kv * residual
        self.vv -= kv * self.pv
        self.pv *= 1 - kp
        self.pp = kp * variance


def assign(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Maximize the number of valid pairs, then minimize cost (up to 3 × 3)."""
    best_key = (0, 0.0)
    best_pairs = []

    def visit(row, used, cost, pairs):
        nonlocal best_key, best_pairs
        if row == len(costs):
            key = (-len(pairs), cost)
            if key < best_key:
                best_key, best_pairs = key, pairs
            return
        for col, value in enumerate(costs[row]):
            if col not in used and isfinite(value):
                visit(row + 1, used | {col}, cost + value, pairs + [(row, col)])
        visit(row + 1, used, cost, pairs)

    visit(0, set(), 0.0, [])
    return best_pairs


@dataclass
class Track:
    id: int
    label: str
    axes: list[KalmanAxis]
    last_seen: float
    hits: int = 1
    box: tuple[int, int, int, int] | None = None
    box_xyz: Vec3 | None = None

    @property
    def position(self) -> Vec3:
        return tuple(axis.position for axis in self.axes)


@dataclass(frozen=True)
class Estimate:
    id: int
    label: str
    xyz: Vec3
    velocity: Vec3 | None
    direction: Vec3 | None
    observed: bool
    forecasts: tuple[Vec3, ...]
    box: tuple[int, int, int, int] | None = None
    box_xyz: Vec3 | None = None


def measurement_variance(measurement: Measurement) -> float:
    return max(0.025, 1.4826 * measurement.depth_mad_m)**2


class Tracker:
    def __init__(self, model: MotionModel, *, max_tracks: int = 3, coast_s: float = 0.30):
        if not 1 <= max_tracks <= 3 or coast_s <= 0:
            raise ValueError("Tracker supports 1 to 3 objects and a positive coast time.")
        self.model = model
        self.max_tracks = max_tracks
        self.coast_s = coast_s
        self.tracks: list[Track] = []
        self.previous_time: float | None = None
        self.next_id = 1

    def step(self, time_s: float, measurements: list[Measurement]) -> list[Estimate]:
        if not isfinite(time_s) or (self.previous_time is not None and time_s <= self.previous_time):
            raise ValueError("Track timestamps must be finite and strictly increasing.")
        dt = 0.0 if self.previous_time is None else time_s - self.previous_time
        self.previous_time = time_s
        self.tracks = [track for track in self.tracks if time_s - track.last_seen <= self.coast_s]
        acceleration = self.model.acceleration_m_s2
        for track in self.tracks:
            for axis, a in zip(track.axes, acceleration):
                axis.predict(dt, a, self.model.process_noise_m2_s4)

        measurements = sorted(
            measurements, key=lambda m: m.detection.confidence, reverse=True
        )[:self.max_tracks]
        pairs = assign([[self._cost(track, m) for m in measurements] for track in self.tracks])
        for row, col in pairs:
            track, m = self.tracks[row], measurements[col]
            for axis, coordinate in zip(track.axes, m.xyz):
                axis.update(coordinate, measurement_variance(m))
            track.last_seen = time_s
            track.hits += 1
            track.box, track.box_xyz = m.detection.bbox, m.xyz
        matched = {col for _, col in pairs}
        for col, m in enumerate(measurements):
            if col in matched or len(self.tracks) == self.max_tracks:
                continue
            self.tracks.append(Track(
                self.next_id, m.detection.label,
                [KalmanAxis(value, pp=measurement_variance(m)) for value in m.xyz], time_s,
                box=m.detection.bbox, box_xyz=m.xyz,
            ))
            self.next_id += 1
        return [self._estimate(track, time_s) for track in self.tracks]

    def _cost(self, track: Track, measurement: Measurement) -> float:
        if track.label != measurement.detection.label or dist(track.position, measurement.xyz) > 0.40:
            return float("inf")
        variance = measurement_variance(measurement)
        d2 = sum(
            (coordinate - axis.position)**2 / (axis.pp + variance)
            for axis, coordinate in zip(track.axes, measurement.xyz)
        )
        return d2 if d2 <= 16.3 else float("inf")

    def _estimate(self, track: Track, time_s: float) -> Estimate:
        xyz = track.position
        velocity = tuple(axis.velocity for axis in track.axes) if track.hits >= 3 else None
        speed = dist(velocity, (0, 0, 0)) if velocity is not None else 0.0
        direction = tuple(v / speed for v in velocity) if velocity is not None and speed >= 0.08 else None
        forecasts = ()
        if velocity is not None:
            forecasts = tuple(
                tuple(p + v * t + a * t * t / 2
                      for p, v, a in zip(xyz, velocity, self.model.acceleration_m_s2))
                for t in HORIZONS
            )
        return Estimate(track.id, track.label, xyz, velocity, direction, track.last_seen == time_s,
                        forecasts, track.box, track.box_xyz)


class Pipeline:
    def __init__(self, detector: Detector, tracker: Tracker):
        self.detector = detector
        self.tracker = tracker

    def step(self, frame: Frame) -> tuple[list[Measurement], list[Estimate]]:
        _, measurements, estimates = self.analyze(frame)
        return measurements, estimates

    def analyze(self, frame: Frame):
        detections = self.detector.detect(frame.bgrx, frame.width, frame.height)
        measurements = [m for d in detections if (m := localize(frame, d)) is not None]
        return detections, measurements, self.tracker.step(frame.depth_ms / 1000, measurements)
