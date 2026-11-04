import io
import unittest
from array import array
from dataclasses import replace
from math import dist, inf

from rgbd.benchmark import evaluate
from rgbd.detection import BallDetector, Detection2D, Measurement, localize
from rgbd.frames import read_frames, write_frame
from rgbd.simulation import simulate
from rgbd.tracking import ConstantVelocity, Pipeline, Tracker, assign


class FrameTests(unittest.TestCase):
    def test_recording_roundtrip(self):
        frame, _ = next(simulate("single"))
        stream = io.BytesIO()
        write_frame(stream, frame)
        stream.seek(0)
        self.assertEqual(list(read_frames(stream)), [frame])

    def test_truncated_frame_rejected(self):
        frame, _ = next(simulate("single"))
        stream = io.BytesIO()
        write_frame(stream, frame)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            list(read_frames(io.BytesIO(stream.getvalue()[:-1])))

    def test_skew_rejected(self):
        frame, _ = next(simulate("single"))
        stream = io.BytesIO()
        write_frame(stream, replace(frame, rgb_ms=26))
        with self.assertRaisesRegex(ValueError, "timestamps"):
            list(read_frames(io.BytesIO(stream.getvalue())))


class LocalizationTests(unittest.TestCase):
    def test_ball_detected_in_color_and_localized(self):
        frame, truth = next(simulate("single"))
        results, tracks = Pipeline(BallDetector(), Tracker(ConstantVelocity())).step(frame)
        self.assertEqual(len(results), 1)
        self.assertLess(dist(results[0].xyz, truth[1].xyz), 0.03)
        self.assertIsNone(tracks[0].direction)

    def test_box_only_plugin_uses_color_aligned_xyz(self):
        frame, truth = next(simulate("single"))

        class BoxDetector:
            def detect(self, bgrx, width, height):
                return [Detection2D((52, 56, 68, 72), 1.0, "proxy")]

        # The raw-depth array is deliberately empty. The plugin's RGB box should
        # only read from the corresponding aligned XYZ values.
        frame = replace(frame, raw_depth_mm=memoryview(array("H", [0]) * (frame.width * frame.height)))
        results, tracks = Pipeline(BoxDetector(), Tracker(ConstantVelocity())).step(frame)
        self.assertEqual(tracks[0].label, "proxy")
        self.assertLess(dist(results[0].xyz, truth[1].xyz), 0.04)

    def test_invalid_and_conflicting_depth_rejected(self):
        frame, _ = next(simulate("single"))
        detection = BallDetector().detect(frame.bgrx, frame.width, frame.height)[0]
        xyz = array("f", [0]) * len(frame.color_xyz_m)
        self.assertIsNone(localize(replace(frame, color_xyz_m=memoryview(xyz)), detection))
        for index, pixel in enumerate(detection.mask_pixels[:24]):
            xyz[pixel * 3 + 2] = 1.0 if index < 12 else 2.5
        self.assertIsNone(localize(replace(frame, color_xyz_m=memoryview(xyz)), detection))


class TrackingTests(unittest.TestCase):
    def test_assignment_prioritizes_coverage(self):
        self.assertEqual(assign([[1, 3], [2, inf]]), [(0, 1), (1, 0)])

    def test_coast_reacquire_then_expire(self):
        tracker = Tracker(ConstantVelocity())
        detection = Detection2D((0, 0, 4, 4), .9, "ball")

        def observed(t):
            return [Measurement(detection, (t, 0, 2), 10, 0)]

        for t in (0, .04, .08, .12):
            tracks = tracker.step(t, observed(t))
        self.assertGreater(tracks[0].direction[0], .99)
        coast = tracker.step(.24, [])[0]
        self.assertFalse(coast.observed)
        self.assertEqual(tracker.step(.32, observed(.32))[0].id, coast.id)
        self.assertEqual(tracker.step(.70, []), [])
        self.assertNotEqual(tracker.step(.75, observed(.75))[0].id, coast.id)

    def test_ballistic_model_improves_matched_synthetic_scenario(self):
        runs = evaluate("ballistic")
        cv = runs["cv"][0]["forecast"]["0.3"]
        gravity = runs["ballistic"][0]["forecast"]["0.3"]
        self.assertEqual(cv["samples"], gravity["samples"])
        self.assertGreater(gravity["samples"], 0)
        self.assertLess(gravity["mean_error_m"], cv["mean_error_m"])


if __name__ == "__main__":
    unittest.main()
