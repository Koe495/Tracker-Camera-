import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from perception import Evaluation, ObstacleMonitor, RGBDPacket, fit_floor, measure_roi
from registration import SDKMapper
from depth_lab import mouse_callback, save_snapshot
import cv2


class PerceptionTests(unittest.TestCase):
    def test_roi_ignores_invalid_depth_and_rejects_low_coverage(self):
        d = np.full((10, 10), 1500, np.uint16)
        d[:4] = 0
        result = measure_roi(d, (0, 0, 10, 10))
        self.assertEqual(result.median_mm, 1500)
        self.assertEqual(result.valid_ratio, .6)
        d[:6] = 0
        self.assertEqual(measure_roi(d, (0, 0, 10, 10)).status, "INSUFFICIENT_DEPTH")
        self.assertIsNone(measure_roi(d, (0, 0, 10, 10)).median_mm)

    def test_bbox_is_clipped_and_empty_rois_are_rejected(self):
        d = np.full((10, 10), 2000, np.uint16)
        self.assertEqual(measure_roi(d, (-4, -3, 10, 10)).bbox, (0, 0, 6, 7))
        self.assertEqual(measure_roi(d, (20, 20, 5, 5)).status, "INVALID_ROI")

    def test_stale_detection_is_not_matched_to_new_depth(self):
        packet = RGBDPacket(10, 300, 303, np.zeros((10, 10, 3), np.uint8),
                            np.full((10, 10), 1200, np.uint16), "simulation")
        with self.assertRaises(ValueError):
            packet.measure_detection(9, (0, 0, 10, 10))
        self.assertEqual(packet.measure_detection(10, (0, 0, 10, 10)).median_mm, 1200)

    def test_calibrated_alignment_respects_sdk_coordinates_and_holes(self):
        mapper = SDKMapper(None)
        depth = np.full((480, 640), 3000, np.uint16)
        depth[:, 80:120] = 1000
        y, x = np.mgrid[:480, :640]
        def color_map(*args):
            mapper.points[:, :, 0] = x + 30
            mapper.points[:, :, 1] = y
            mapper.points[:, :, 2:] = 0
            return 0
        def xyz_map(*args):
            mapper.xyz4[:] = 1
            return 0
        mapper.map_color, mapper.map_xyz = color_map, xyz_map
        aligned, xyz = mapper.align(depth)
        self.assertEqual(aligned[100, 60], 1000)  # original pixel 90, NOT RGB pixel 60
        self.assertEqual(aligned[100, 100], 3000)
        self.assertTrue(np.all(aligned[:, -30:] == 0))  # outside depth camera field of view
        self.assertEqual(xyz.shape, (480, 640, 3))

    @staticmethod
    def scene(depth=3000):
        d = np.full((120, 180), depth, np.uint16)
        y, x = np.mgrid[:120, :180]
        z = d.astype(np.float32) / 1000
        xyz = np.stack(((x - 90) / 150 * z, (60 - y) / 150 * z, z), -1)
        return d, xyz

    def test_unknown_is_immediate_and_near_requires_confirmation(self):
        d, xyz = self.scene()
        monitor = ObstacleMonitor()
        d[30:80, 70:100] = 900
        for _ in range(2):
            self.assertEqual(monitor.update(d, xyz)[1].status, "CHECKING")
        self.assertEqual(monitor.update(d, xyz)[1].status, "DANGER")
        d[:] = 0
        self.assertTrue(all(z.status == "UNKNOWN" for z in monitor.update(d, xyz)))

    def test_a_single_bad_pixel_does_not_trigger_an_obstacle(self):
        d, xyz = self.scene()
        d[50, 80] = 800
        monitor = ObstacleMonitor()
        for _ in range(3):
            zones = monitor.update(d, xyz)
        self.assertTrue(all(z.status == "NO_NEAR_SURFACE" for z in zones))

    def test_floor_fit_accepts_ground_and_rejects_vertical_wall(self):
        y, x = np.mgrid[:240, :320]
        xyz = np.stack(((x - 160) / 100, np.full_like(x, -1, dtype=float),
                        1 + y / 100), -1).astype(np.float32)
        floor = fit_floor(xyz)
        self.assertIsNotNone(floor)
        self.assertAlmostEqual(float(floor[1]), 1., places=3)
        self.assertAlmostEqual(float(floor[3]), 1., places=3)
        xyz[:, :, 1] = (120 - y) / 100
        xyz[:, :, 2] = 2
        self.assertIsNone(fit_floor(xyz))

    def test_floor_mask_preserves_object_above_floor(self):
        d, xyz = self.scene(900)
        xyz[:, :, 1] = -1
        xyz[30:80, 70:100, 1] = -.4
        monitor = ObstacleMonitor()
        monitor.floor = np.array([0, 1, 0, 1], np.float32)
        for _ in range(3):
            zones = monitor.update(d, xyz)
        self.assertEqual(zones[1].status, "DANGER")
        self.assertEqual(zones[0].status, "NO_NEAR_SURFACE")

    def test_evaluation_reports_error_jitter_and_simulation_source(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Evaluation(folder, 1500, 3, "simulation")
            for i, value in enumerate((1490, 1500, 1510)):
                depth = np.full((10, 10), value, np.uint16)
                p = RGBDPacket(i, i * 33, i * 33, np.zeros((10, 10, 3)), depth, "simulation")
                finished = run.add(p, measure_roi(depth, (0, 0, 10, 10)))
            self.assertTrue(finished)
            path = run.save("test")
            summary = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(summary["source"], "simulation")
            self.assertEqual(summary["mean_error_mm"], 0)
            self.assertAlmostEqual(summary["mae_mm"], 20 / 3)
            self.assertAlmostEqual(summary["temporal_std_mm"], np.std([1490, 1500, 1510]))
            with path.open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_mouse_roi_uses_same_rgb_grid_on_both_panels(self):
        state = dict(recording=False, drag=None)
        mouse_callback(cv2.EVENT_LBUTTONDOWN, 680, 96, 0, state)
        mouse_callback(cv2.EVENT_LBUTTONUP, 710, 126, 0, state)
        self.assertEqual(state["roi"], (20, 20, 31, 31))
        state["recording"] = True
        mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, state)
        self.assertEqual(state["roi"], (20, 20, 31, 31))

    def test_snapshot_keeps_native_and_aligned_depth_separate(self):
        native = np.full((10, 10), 2000, np.uint16)
        aligned = np.full((10, 10), 1900, np.uint16)
        packet = RGBDPacket(42, 10, 12, np.zeros((10, 10, 3), np.uint8), aligned, "simulation")
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(Path(directory), packet, native, np.zeros((10, 10, 3)),
                                 measure_roi(aligned, (0, 0, 10, 10)), [], None)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["native_depth_mm"], native)
                np.testing.assert_array_equal(saved["depth_aligned_mm"], aligned)
                self.assertEqual(json.loads(str(saved["metadata_json"]))["frame_id"], 42)

    def test_main_lab_switch_delegates_arguments(self):
        from main import main
        with patch("depth_lab.main", return_value=0) as lab:
            self.assertEqual(main(["--lab", "--demo"]), 0)
            lab.assert_called_once_with(["--demo"])


if __name__ == "__main__":
    unittest.main()
