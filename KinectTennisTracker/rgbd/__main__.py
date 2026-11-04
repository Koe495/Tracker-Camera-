import argparse
import json
import subprocess
from importlib import import_module
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile
from time import perf_counter

from .benchmark import evaluate
from .detection import BallDetector
from .frames import MAX_DEPTH_M, MIN_DEPTH_M, read_frames
from .report import save_report
from .simulation import SCENARIOS, simulate
from .tracking import Ballistic, ConstantVelocity, Pipeline, Tracker

BRIDGE = Path(__file__).resolve().parents[1] / "kinect" / "KinectBridge.exe"
MODELS = {"cv": ConstantVelocity, "ballistic": Ballistic}


def load_factory(spec):
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError("Use a local plugin in module:factory format.")
    return getattr(import_module(module), name)


def capture(out: Path, count: int, bridge: Path):
    if not 1 <= count <= 600:
        raise ValueError("Choose 1–600 frames. A 640×480 frame is about 5.5 MB.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=out.parent, suffix=".part", delete=False) as file:
        temp = Path(file.name)
        try:
            subprocess.run([str(bridge.resolve()), "stream", str(count)], stdout=file,
                           check=True, timeout=20 + count / 3)
            file.close()
            with temp.open("rb") as stream:
                accepted = sum(1 for _ in read_frames(stream))
            if accepted != count:
                raise ValueError(f"Expected {count} frame pairs, received {accepted}.")
            temp.replace(out)
        finally:
            file.close()
            temp.unlink(missing_ok=True)
    return {"frames": count, "bytes": out.stat().st_size, "recording": str(out)}


def summarize_capture(frames, pipeline):
    rows, skew, valid_raw, valid_aligned, timings = [], [], [], [], []
    frame = None
    for frame in frames:
        start = perf_counter()
        measurements, tracks = pipeline.step(frame)
        timings.append((perf_counter() - start) * 1000)
        rows.append((frame.depth_ms, measurements, tracks))
        skew.append(abs(frame.rgb_ms - frame.depth_ms))
        valid_raw.append(sum(
            MIN_DEPTH_M * 1000 <= value <= MAX_DEPTH_M * 1000 for value in frame.raw_depth_mm
        ) / (frame.depth_width * frame.depth_height))
        valid_aligned.append(sum(
            MIN_DEPTH_M <= value <= MAX_DEPTH_M for value in frame.color_xyz_m[2::3]
        ) / (frame.width * frame.height))
    if frame is None:
        raise ValueError("The recording contains no frame pairs.")
    duration = (rows[-1][0] - rows[0][0]) / 1000
    metrics = {
        "frames": len(rows), "color_size": (frame.width, frame.height),
        "depth_size": (frame.depth_width, frame.depth_height),
        "accepted_fps": (len(rows) - 1) / duration if duration else None,
        "rgb_depth_skew_median_ms": median(skew), "rgb_depth_skew_max_ms": max(skew),
        "depth_valid_m": [MIN_DEPTH_M, MAX_DEPTH_M],
        "raw_depth_valid_fraction": mean(valid_raw), "aligned_depth_valid_fraction": mean(valid_aligned),
        "frames_with_3d_measurements": sum(bool(measurements) for _, measurements, _ in rows),
        "pipeline_mean_ms": mean(timings),
    }
    return metrics, rows, frame


def run_benchmark(out, seed):
    summary = {"source": "simulation", "seed": seed, "scenarios": {}}
    for scenario in SCENARIOS:
        runs = evaluate(scenario, seed)
        summary["scenarios"][scenario] = {key: metrics for key, (metrics, _) in runs.items()}
        key = "ballistic" if scenario == "ballistic" else "cv"
        metrics, rows = runs[key]
        frame = None
        for frame, _ in simulate(scenario, seed):
            pass
        save_report(out / scenario, rows, title=f"RGB-D Motion Lab · {scenario}",
                    source="GIẢ LẬP", frame=frame, metrics=metrics)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"benchmark": str(out / "benchmark.json"), "dashboard": str(out / "crossing" / "dashboard.html")}


def main():
    parser = argparse.ArgumentParser(description="RGB-D 3D Object Motion Tracking & Prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="List Kinect v1 sensor status.")
    probe.add_argument("--bridge", type=Path, default=BRIDGE)
    record = commands.add_parser("capture", help="Record paired RGB, depth and calibrated XYZ.")
    record.add_argument("recording", type=Path)
    record.add_argument("--frames", type=int, default=30)
    record.add_argument("--bridge", type=Path, default=BRIDGE)
    replay = commands.add_parser("replay", help="Process a Kinect recording and build a 3D report.")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--out", type=Path, default=Path("replay-report"))
    replay.add_argument("--model", default="cv", help="cv, ballistic or module:factory")
    replay.add_argument("--detector", default="ball", help="ball (HSV), tennis, uav or module:factory")
    replay.add_argument("--stride", type=int, choices=range(1, 5), default=2)
    replay.add_argument("--hue-min", type=float, default=8)
    replay.add_argument("--hue-max", type=float, default=42)
    bench = commands.add_parser("bench", help="Run seeded benchmarks and export 3D dashboards.")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--out", type=Path, default=Path("sim-report"))
    args = parser.parse_args()
    try:
        if args.command == "probe":
            return subprocess.run([str(args.bridge.resolve()), "probe"], timeout=12).returncode
        if args.command == "capture":
            result = capture(args.recording, args.frames, args.bridge)
        elif args.command == "bench":
            result = run_benchmark(args.out, args.seed)
        else:
            if not all(0 <= hue <= 360 for hue in (args.hue_min, args.hue_max)):
                raise ValueError("Hue must be between 0 and 360 degrees.")
            if args.detector == "ball":
                detector = BallDetector(args.hue_min, args.hue_max, stride=args.stride)
            elif args.detector in ("tennis", "uav"):
                from .yolo import DETECTORS

                detector = DETECTORS[args.detector]()
            else:
                detector = load_factory(args.detector)()
            model = (MODELS[args.model] if args.model in MODELS else load_factory(args.model))()
            pipeline = Pipeline(detector, Tracker(model))
            with args.recording.open("rb") as stream:
                result, rows, frame = summarize_capture(read_frames(stream), pipeline)
            result.update(detector=args.detector, motion_model=args.model)
            save_report(args.out, rows, title="RGB-D Motion Lab · Kinect v1",
                        source="BẢN GHI RGB-D", frame=frame, metrics=result)
            (args.out / "capture.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, ImportError, AttributeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"RGB-D: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
