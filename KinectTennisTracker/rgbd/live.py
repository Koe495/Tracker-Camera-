import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from threading import Condition, Thread
from time import monotonic, perf_counter

import cv2
import numpy as np

from .__main__ import BRIDGE, MODELS
from .frames import MAX_DEPTH_M, MIN_DEPTH_M, read_frames
from .preview import WINDOW_H, WINDOW_W, render_preview
from .projection import read_projection
from .report import save_report
from .tennis import DEFAULT_MODEL, TennisDetector
from .tracking import HORIZONS, Pipeline, Tracker

WINDOW = "Kinect Tennis Tracker"


class LiveFrames:
    """Drain SDK output continuously and give inference the newest complete frame."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.condition = Condition()
        self.latest = None
        self.error = None
        self.finished = False
        self.received = 0
        self.dropped = 0
        self.projection = None

    def __enter__(self):
        self.process = subprocess.Popen(
            [str(self.bridge.resolve()), "live-project"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _read(self):
        try:
            self.projection = read_projection(self.process.stdout)
            for frame in read_frames(self.process.stdout):
                with self.condition:
                    self.received += 1
                    self.dropped += int(self.latest is not None)
                    self.latest = frame
                    self.condition.notify()
        except (OSError, ValueError) as error:
            self.error = error
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify()

    def take(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.latest is not None or self.finished, timeout=12):
                raise TimeoutError("Timed out waiting for Kinect RGB-D.")
            if self.latest is not None:
                frame, self.latest = self.latest, None
                return frame
            raise OSError(f"Kinect stream stopped (exit {self.process.wait(timeout=5)})."
                          + (f" {self.error}" if self.error else ""))

    def __exit__(self, *_):
        try:
            self.process.stdin.write(b"stop\n")
            self.process.stdin.flush()
        except OSError:
            pass
        finally:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.reader.join()
            self.process.stdout.close()


def preview_frame(frame, detections, measurements, tracks, fps, projection):
    cv2.imshow(WINDOW, render_preview(frame, detections, measurements, tracks, fps, projection))
    key = cv2.waitKey(1) & 0xFF
    try:
        return key not in (ord("q"), 27) and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def run_live(args, detector):
    if args.seconds is not None and (not isfinite(args.seconds) or not 0 < args.seconds <= 120):
        raise ValueError("Choose a duration between 0 and 120 seconds.")
    if (args.headless or args.out) and args.seconds is None:
        raise ValueError("--headless and --out require --seconds.")
    pipeline = Pipeline(detector, Tracker(MODELS[args.motion]()))
    frames = rgb_hits = depth_hits = tracked_frames = forecast_frames = 0
    skew, rows = Counter(), []
    coverage_sum = time_sum_ms = 0.0
    first = monotonic()
    latest_tracks = []
    try:
        with LiveFrames(args.bridge) as source:
            while True:
                frame = source.take()
                if frames == 0:
                    first = monotonic()
                    if not args.headless:
                        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(WINDOW, round(WINDOW_W * 660 / WINDOW_H), 660)
                start = perf_counter()
                detections, measurements, tracks = pipeline.analyze(frame)
                time_sum_ms += (perf_counter() - start) * 1000
                frames += 1
                rgb_hits += bool(detections)
                depth_hits += bool(measurements)
                tracked_frames += bool(tracks)
                forecast_frames += any(t.forecasts for t in tracks)
                if tracks:
                    latest_tracks = [asdict(t) for t in tracks]
                skew[abs(frame.rgb_ms - frame.depth_ms)] += 1
                points = np.asarray(frame.color_xyz_m).reshape(-1, 3)[:, 2]
                coverage_sum += float(np.mean((MIN_DEPTH_M <= points) & (points <= MAX_DEPTH_M)))
                elapsed = monotonic() - first
                if args.out:
                    rows.append((frame.depth_ms, measurements, tracks))
                if not args.headless and not preview_frame(frame, detections, measurements, tracks,
                                                          frames / max(.001, elapsed), source.projection):
                    break
                if args.seconds is not None and elapsed >= args.seconds:
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
    def skew_rank(rank):
        for value, count in sorted(skew.items()):
            if rank < count:
                return value
            rank -= count

    metrics = {
        "source": "kinect_live", "motion_model": args.motion, "forecast_horizons_s": HORIZONS,
        "received_rgbd_pairs": source.received, "processed_frames": frames,
        "superseded_frames": source.dropped, "processed_fps": frames / max(.001, elapsed),
        "pipeline_mean_ms": time_sum_ms / frames,
        "rgb_depth_skew_median_ms": (skew_rank((frames - 1) // 2) + skew_rank(frames // 2)) / 2,
        "rgb_depth_skew_max_ms": max(skew), "aligned_depth_valid_fraction": coverage_sum / frames,
        "projection_fit_p95_px": source.projection.calibration_p95_px if source.projection else None,
        "frames_with_2d_ball": rgb_hits, "frames_with_3d_ball": depth_hits,
        "frames_with_tracks": tracked_frames, "frames_with_forecasts": forecast_frames,
        "last_nonempty_tracks": latest_tracks,
    }
    if args.out:
        save_report(args.out, rows, title="Kinect Tennis Tracker · Quỹ đạo 3D",
                    source="KINECT THẬT", frame=frame, metrics=metrics)
        (args.out / "capture.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tennis ball detection and Kinect 3D tracking")
    commands = parser.add_subparsers(dest="command", required=True)
    camera = commands.add_parser("camera")
    camera.add_argument("--bridge", type=Path, default=BRIDGE)
    camera.add_argument("--motion", choices=MODELS, default="cv")
    camera.add_argument("--seconds", type=float)
    camera.add_argument("--headless", action="store_true")
    camera.add_argument("--out", type=Path)
    image = commands.add_parser("image")
    image.add_argument("path", type=Path)
    for command in (camera, image):
        command.add_argument("--weights", type=Path, default=DEFAULT_MODEL)
        command.add_argument("--confidence", type=float, default=.25)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--imgsz", type=int, choices=(640, 960, 1280), default=640)
    args = parser.parse_args()
    try:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        detector = TennisDetector(args.weights, args.confidence, args.threads, args.imgsz)
        if args.command == "camera":
            result = run_live(args, detector)
        else:
            image = cv2.imdecode(np.fromfile(args.path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {args.path}.")
            result = [{"bbox": d.bbox, "confidence": d.confidence, "label": d.label}
                      for d in detector.detect_image(image)]
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Kinect Tennis Tracker: {error}\n")


if __name__ == "__main__":
    main()
