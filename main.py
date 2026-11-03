"""Run: python main.py   |   Preview without Kinect: python main.py --demo"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependencies. Run: python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)

from kinect_v1 import KinectError, KinectV1, diagnose
from vision import DemoSource, Pairer, dashboard, depth_image

WINDOW = "Kinect 1517 - Color and Depth"


def save_pair(folder, color, depth, near, far, demo):
    target = folder / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target.mkdir(parents=True, exist_ok=False)
    for name, array in (("color.png", color.data), ("depth_mm.png", depth.data),
                        ("depth_color.png", depth_image(depth.data, near, far))):
        # imencode + tofile supports Vietnamese Windows paths.
        ok, encoded = cv2.imencode(".png", array)
        if not ok:
            raise OSError(f"Failed to encode {name}")
        encoded.tofile(str(target / name))
    (target / "metadata.json").write_text(json.dumps({
        "source": "simulation" if demo else "Kinect model 1517 / SDK 1.8",
        "depth_unit": "millimetres", "invalid_depth": 0,
        "color_timestamp_ms": color.timestamp_ms, "depth_timestamp_ms": depth.timestamp_ms,
        "color_frame": color.number, "depth_frame": depth.number,
        "depth_range_mm": [near, far], "spatially_registered": False,
    }, indent=2), encoding="utf-8")
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Kinect model 1517: live RGB + metric depth")
    parser.add_argument("--demo", action="store_true", help="Explicit synthetic source, no hardware")
    parser.add_argument("--diagnose", action="store_true", help="Check SDK runtime and sensor count")
    parser.add_argument("--near-mode", action="store_true", help="Enable Kinect for Windows near mode")
    parser.add_argument("--min-depth", type=int, default=800, metavar="MM")
    parser.add_argument("--max-depth", type=int, default=4000, metavar="MM")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds without a valid frame pair")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N pairs (0 = unlimited)")
    parser.add_argument("--headless", action="store_true", help="No GUI; requires --frames")
    parser.add_argument("--preview", type=Path, help="Save last dashboard as PNG")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "captures")
    args = parser.parse_args(argv)
    if not 0 < args.min_depth < args.max_depth <= 4094:
        parser.error("Depth range must satisfy 0 < min-depth < max-depth <= 4094 mm.")
    if args.timeout <= 0 or args.frames < 0:
        parser.error("Timeout must be positive; frames cannot be negative.")
    if args.headless and args.frames == 0 and not args.diagnose:
        parser.error("--headless requires --frames N to limit the run.")
    return args


def run(args):
    if args.diagnose:
        print(diagnose())
        return 0
    source = DemoSource() if args.demo else KinectV1(args.near_mode)
    pairer = Pairer(tolerance_ms=40)
    if args.demo:
        print("SIMULATION: synthetic RGB and depth; no Kinect connection.")
    else:
        print("Opening Kinect model 1517 using SDK 1.8...")
    print("Q/ESC: quit | G: depth color/gray | S: save RGB and 16-bit depth PNG")
    count, fps, gray, canvas = 0, 0.0, False, None
    status, status_until = "", 0.0
    try:
        with source:
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(WINDOW, 1280, 632)
                waiting = np.zeros((646, 1308, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for Kinect RGB + depth...", (250, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(WINDOW, waiting)
            last_pair = time.monotonic()
            fps_started, fps_frames = last_pair, 0
            current = None
            while True:
                color = source.read("color")
                depth = source.read("depth")
                pair = pairer.add(color, depth)
                now = time.monotonic()
                if pair is not None:
                    current = pair
                    last_pair = now
                    count += 1
                    fps_frames += 1
                    if now - fps_started >= 0.5:
                        fps = fps_frames / (now - fps_started)
                        fps_started, fps_frames = now, 0
                    canvas = dashboard(*pair, args.min_depth, args.max_depth, fps,
                                       args.demo, gray, status if now < status_until else "")
                    if not args.headless:
                        cv2.imshow(WINDOW, canvas)
                    if args.frames and count >= args.frames:
                        break
                if now - last_pair > args.timeout:
                    raise KinectError(f"No matching RGB/depth pair for {args.timeout:g}s. "
                                      "Check camera power, USB and SDK 1.8 drivers.")
                if not args.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("g"):
                        gray = not gray
                    if key == ord("s") and current is not None:
                        try:
                            target = save_pair(args.output_dir, *current, args.min_depth,
                                               args.max_depth, args.demo)
                            print(f"Saved: {target}")
                            status = "Saved RGB + depth_mm.png (uint16 millimetres)"
                        except OSError as exc:
                            print(f"Cannot save: {exc}", file=sys.stderr)
                            status = "Save failed - see console"
                        status_until = time.monotonic() + 3
                time.sleep(0.002)
        if args.preview and canvas is not None:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", canvas)
            if not ok:
                raise OSError("Cannot encode preview")
            encoded.tofile(str(args.preview))
        print(f"Stopped. Received {count} RGB/depth pairs.")
        return 0
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--lab" in argv:
        from depth_lab import main as lab_main
        argv.remove("--lab")
        return lab_main(argv)
    try:
        return run(parse_args(argv))
    except (KinectError, OSError, cv2.error) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
