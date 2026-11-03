"""Next milestone: calibrated RGB-D measurement, evaluation and surface alerts."""
from contextlib import ExitStack
from datetime import datetime
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from kinect_v1 import Frame, KinectError, KinectV1
from perception import Evaluation, ObstacleMonitor, RGBDPacket, measure_roi
from registration import DemoMapper, SDKMapper
from vision import DemoSource, Pairer, depth_image, text

WINDOW = "Kinect 1517 - RGB-D Measurement Lab"


class AudioAlerts:
    """One shared nonblocking alert output; off unless --audio is supplied."""
    def __init__(self, enabled):
        self.enabled = enabled and sys.platform == "win32"
        self.queue = queue.Queue(maxsize=1)
        self.last = 0
        self.thread = None

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        return self

    def _worker(self):
        import winsound
        while True:
            value = self.queue.get()
            if value is None:
                return
            try:
                winsound.Beep(1200 if value == "DANGER" else 800, 100)
            except RuntimeError:
                pass

    def update(self, zones):
        priority = next((s for s in ("DANGER", "NEAR") if any(z.status == s for z in zones)), None)
        now = time.monotonic()
        if self.enabled and priority and now - self.last >= 3:
            try:
                self.queue.put_nowait(priority)
                self.last = now
            except queue.Full:
                pass

    def __exit__(self, *_):
        if self.thread:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(None)
            self.thread.join(timeout=.25)


def mouse_callback(event, x, y, flags, state):
    if state.get("recording"):
        return  # freeze ROI while collecting an evaluation run
    offset = 8 if 8 <= x < 648 else 660 if 660 <= x < 1300 else None
    if event == cv2.EVENT_LBUTTONDOWN and offset is not None and 76 <= y < 556:
        state["drag"] = (x - offset, y - 76, offset)
        state["roi"] = (max(0, x - offset - 5), max(0, y - 81), 11, 11)
    elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and state.get("drag"):
        sx, sy, origin = state["drag"]
        ex, ey = np.clip(x - origin, 0, 639), np.clip(y - 76, 0, 479)
        if abs(ex - sx) > 3 or abs(ey - sy) > 3:
            state["roi"] = (int(min(sx, ex)), int(min(sy, ey)),
                            int(abs(ex - sx)) + 1, int(abs(ey - sy)) + 1)
        if event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None


def render(packet, measure, zones, args, fps, floor_ready, gray, overlay, status):
    canvas = np.full((820, 1308, 3), (24, 20, 17), np.uint8)
    text(canvas, "KINECT 1517 / RGB-D MEASUREMENT", (16, 30), .78)
    text(canvas, "SIMULATION - NO CAMERA DATA" if args.demo else "LIVE / SDK CALIBRATION", (810, 29), .6,
         (60, 180, 250) if args.demo else (100, 230, 120))
    text(canvas, "RGB - drag to measure a point / region", (12, 62))
    delta = abs(packet.color_timestamp_ms - packet.depth_timestamp_ms)
    text(canvas, f"DEPTH ALIGNED TO RGB | pair delta {delta} ms", (662, 62))
    colored = depth_image(packet.depth_aligned_mm, args.min_depth, args.max_depth, gray)
    color = packet.color_bgr.copy()
    if overlay:
        valid = packet.depth_aligned_mm > 0
        blended = cv2.addWeighted(color, .55, colored, .45, 0)
        color[valid] = blended[valid]
    x, y, w, h = measure.bbox
    for image in (color, colored):
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)
    canvas[76:556, 8:648] = color
    canvas[76:556, 660:1300] = colored
    result = f"{measure.median_mm / 1000:.3f} m" if measure.median_mm is not None else "N/A"
    spatial = f"{measure.std_mm:.1f} mm" if measure.std_mm is not None else "N/A"
    text(canvas, f"ROI median: {result} | valid: {measure.valid_ratio:.1%} | spatial std: {spatial}", (14, 584), .61)
    text(canvas, f"{fps:.1f} FPS | {measure.status} | bbox (x,y,w,h): {measure.bbox}", (14, 610), .54)
    for i, zone in enumerate(zones):
        origin = i * 436
        ink = ((70, 80, 250) if zone.status == "DANGER" else (40, 180, 255)
               if zone.status in ("NEAR", "CHECKING") else (180, 190, 200))
        distance = f"{zone.distance_mm / 1000:.2f} m" if zone.distance_mm else "--"
        cv2.rectangle(canvas, (origin + 8, 628), (origin + 428, 688), ink, 1)
        text(canvas, f"{zone.direction}: {zone.status} | {distance}", (origin + 18, 653), .52, ink)
        text(canvas, f"Native depth coverage: {zone.coverage:.0%}", (origin + 18, 677), .46)
    text(canvas, "Floor calibrated - keep camera fixed" if floor_ready else
         "Floor NOT calibrated: floor can trigger alerts. Point at floor and press F.", (14, 714), .55)
    text(canvas, "Observed surfaces only; no path-clear or drop-off detection. Directions use camera view.", (14, 739), .48)
    text(canvas, "Drag: ROI | O: overlay | G: gray | F: fit floor | C: clear floor | S: snapshot | Q/Esc: quit", (14, 766), .49)
    text(canvas, status or "1/2/3/4: reference 1/1.5/2/3 m | E: collect evaluation samples", (14, 799), .51)
    return canvas


def save_snapshot(folder, packet, native_depth, xyz, measurement, zones, floor):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / datetime.now().strftime("rgbd_%Y%m%d_%H%M%S_%f.npz")
    metadata = dict(frame_id=packet.frame_id, color_timestamp_ms=packet.color_timestamp_ms,
                    depth_timestamp_ms=packet.depth_timestamp_ms, source=packet.source,
                    depth_unit="millimetres", xyz_unit="metres", aligned_to="RGB",
                    roi=measurement.__dict__, zones=[z.__dict__ for z in zones])
    np.savez_compressed(target, color_bgr=packet.color_bgr, native_depth_mm=native_depth,
                        depth_aligned_mm=packet.depth_aligned_mm, xyz_depth_m=xyz,
                        floor_plane=np.array([]) if floor is None else floor,
                        metadata_json=np.array(json.dumps(metadata)))
    return target


def arguments(argv=None):
    p = argparse.ArgumentParser(description="Kinect RGB-D measurement and near-surface prototype")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--near-mode", action="store_true")
    p.add_argument("--min-depth", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=4000)
    p.add_argument("--warning-mm", type=int, default=1500)
    p.add_argument("--danger-mm", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--roi", type=int, nargs=4, default=(290, 210, 60, 60), metavar=("X", "Y", "W", "H"))
    p.add_argument("--reference-mm", type=int, default=1500)
    p.add_argument("--measure-frames", type=int, default=60)
    p.add_argument("--evaluate", action="store_true", help="Start collecting immediately")
    p.add_argument("--audio", action="store_true", help="Rate-limited short alert tones")
    p.add_argument("--preview", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "measurements")
    a = p.parse_args(argv)
    if not 0 < a.min_depth < a.max_depth <= 4094:
        p.error("Require 0 < min-depth < max-depth <= 4094 mm")
    if not a.min_depth <= a.danger_mm < a.warning_mm <= a.max_depth:
        p.error("Require min-depth <= danger-mm < warning-mm <= max-depth")
    if a.frames < 0 or a.timeout <= 0 or a.measure_frames < 1 or a.reference_mm <= 0:
        p.error("Invalid frame count, timeout or reference distance")
    if a.headless and a.frames == 0:
        p.error("--headless requires --frames N")
    if a.evaluate and a.frames and a.frames < a.measure_frames:
        p.error("--frames must be at least --measure-frames for a full evaluation")
    x, y, w, h = a.roi
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > 640 or y + h > 480:
        p.error("ROI must be within the 640x480 image")
    return a


def run(args):
    monitor = ObstacleMonitor(args.warning_mm, args.danger_mm)
    state = dict(roi=tuple(args.roi), drag=None, recording=args.evaluate)
    source_name = "simulation" if args.demo else "kinect1517"
    evaluation = Evaluation(args.output_dir, args.reference_mm, args.measure_frames, source_name) if args.evaluate else None
    reference = args.reference_mm
    pairer = Pairer(40)
    count, fps, fps_count, gray, overlay, canvas = 0, 0., 0, False, False, None
    status = f"Reference {reference / 1000:g} m. Press E to collect {args.measure_frames} samples."
    try:
        with ExitStack() as stack:
            source = stack.enter_context(DemoSource() if args.demo else KinectV1(args.near_mode))
            mapper = stack.enter_context(DemoMapper() if args.demo else SDKMapper(source.dll))
            audio = stack.enter_context(AudioAlerts(args.audio))
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(WINDOW, 1280, 802)
                cv2.setMouseCallback(WINDOW, mouse_callback, state)
                cv2.imshow(WINDOW, np.zeros((820, 1308, 3), np.uint8))
            print(f"RGB-D Lab: {source_name}; calibrated mapping; no semantic detector loaded.")
            last_pair = fps_started = time.monotonic()
            latest = None
            while True:
                pair = pairer.add(source.read("color"), source.read("depth"))
                now = time.monotonic()
                if pair is not None:
                    color, depth = pair
                    aligned, xyz = mapper.align(depth.data)
                    packet = RGBDPacket(color.number, color.timestamp_ms, depth.timestamp_ms,
                                        color.data, aligned, source_name)
                    measurement = measure_roi(aligned, state["roi"], args.min_depth, args.max_depth)
                    zones = monitor.update(depth.data, xyz, args.min_depth, args.max_depth)
                    audio.update(zones)
                    count += 1
                    fps_count += 1
                    if now - fps_started >= .5:
                        fps = fps_count / (now - fps_started)
                        fps_started, fps_count = now, 0
                    if evaluation:
                        finished = evaluation.add(packet, measurement)
                        status = f"Recording {len(evaluation.rows)}/{args.measure_frames} | reference {reference / 1000:g} m"
                        if finished:
                            target = evaluation.save(datetime.now().strftime("eval_%Y%m%d_%H%M%S_%f"))
                            print(f"Evaluation saved: {target}")
                            status = f"Saved {target.name} | reference {reference / 1000:g} m"
                            evaluation, state["recording"] = None, False
                    canvas = render(packet, measurement, zones, args, fps,
                                    monitor.floor is not None, gray, overlay, status)
                    latest = (packet, depth.data, xyz, measurement, zones)
                    last_pair = time.monotonic()
                    if not args.headless:
                        cv2.imshow(WINDOW, canvas)
                    if args.frames and count >= args.frames:
                        break
                if time.monotonic() - last_pair > args.timeout:
                    raise KinectError("No valid RGB/depth pair. Camera disconnected or in use.")
                if not args.headless:
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("g"):
                        gray = not gray
                    elif key == ord("o"):
                        overlay = not overlay
                    elif key == ord("f") and latest:
                        ok = monitor.calibrate_floor(latest[2])
                        status = "Floor fitted. Keep camera fixed; press C after moving it." if ok else "Floor fit rejected. Show more floor; keep camera roughly level."
                    elif key == ord("c"):
                        monitor.floor = None
                        for history in monitor.histories.values():
                            history.clear()
                        status = "Floor model cleared."
                    elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and not evaluation:
                        reference = (1000, 1500, 2000, 3000)[key - ord("1")]
                        status = f"Reference {reference / 1000:g} m. Place target at measured distance; press E."
                    elif key == ord("e") and not evaluation and latest:
                        evaluation = Evaluation(args.output_dir, reference, args.measure_frames, source_name)
                        state["recording"] = True
                    elif key == ord("s") and latest:
                        target = save_snapshot(args.output_dir, *latest, monitor.floor)
                        status = f"Saved {target.name} (raw + aligned depth + XYZ + metadata)"
                        print(status)
                time.sleep(.002)
            if evaluation and evaluation.rows:
                target = evaluation.save(datetime.now().strftime("partial_%Y%m%d_%H%M%S_%f"))
                print(f"Partial evaluation saved: {target}")
                evaluation = None
            if latest:
                print(json.dumps(dict(source=source_name, pairs=count, roi=latest[3].__dict__,
                                     zones=[z.__dict__ for z in latest[4]])))
        if args.preview and canvas is not None:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            ok, image = cv2.imencode(".png", canvas)
            if not ok:
                raise OSError("Preview encode failed")
            image.tofile(str(args.preview))
        return 0
    finally:
        if evaluation and evaluation.rows:
            target = evaluation.save(datetime.now().strftime("partial_%Y%m%d_%H%M%S_%f"))
            print(f"Partial evaluation saved: {target}")
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    try:
        return run(arguments(argv))
    except (KinectError, OSError, ValueError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
