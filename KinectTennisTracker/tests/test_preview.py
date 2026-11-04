import unittest
from dataclasses import replace

import numpy as np

from rgbd.detection import BallDetector
from rgbd.preview import (
    APPROACHING, BACKGROUND, PANEL_W, RECEDING, STEADY_DEPTH, VIEW_H, WINDOW_H, WINDOW_W,
    _depth_color, _depth_opacity, _draw_dashed_box, colorize_depth, panel_origin, render_preview,
)
from rgbd.projection import ColorProjection, forecast_boxes
from rgbd.simulation import FOCAL, HEIGHT, WIDTH, simulate
from rgbd.tracking import ConstantVelocity, Estimate, Pipeline, Tracker

SYNTHETIC_PROJECTION = ColorProjection(np.array(
    [[FOCAL, 0, WIDTH / 2, 0], [0, -FOCAL, HEIGHT / 2, 0], [0, 0, 1, 0]]), 0)


class PreviewTests(unittest.TestCase):
    def test_opacity_uses_axial_speed_magnitude_with_a_cap(self):
        self.assertAlmostEqual(_depth_opacity(0, .5), .35)
        self.assertAlmostEqual(_depth_opacity(-.5, .5), .65)
        self.assertAlmostEqual(_depth_opacity(.1, .1), .65)
        self.assertAlmostEqual(_depth_opacity(1, .5), .95)
        self.assertAlmostEqual(_depth_opacity(-2, .5), .95)

    def test_faster_box_has_stronger_color_over_same_rgb_background(self):
        base = np.full((32, 48, 3), 40, dtype=np.uint8)
        slow, fast = base.copy(), base.copy()
        box = (-4, 8, 28, 24)
        _draw_dashed_box(slow, box, APPROACHING, _depth_opacity(-.1, .5))
        _draw_dashed_box(fast, box, APPROACHING, _depth_opacity(-1, .5))
        self.assertGreater(int(fast[8, 2, 2]), int(slow[8, 2, 2]))
        np.testing.assert_array_equal(slow[16, 14], base[16, 14])
        np.testing.assert_array_equal(fast[16, 14], base[16, 14])

    def test_depth_colormap_marks_missing_and_out_of_range(self):
        values = np.array([[0, np.nan, np.inf, .5, .8, 2, 4, 4.1]], dtype=np.float32)
        colors = colorize_depth(values)[0]
        for index in (0, 1, 2, 3, 7):
            np.testing.assert_array_equal(colors[index], BACKGROUND)
        self.assertEqual(len({tuple(color) for color in colors[4:7]}), 3)

    def test_dashboard_renders_four_synchronized_panels(self):
        tracker = Pipeline(BallDetector(stride=1), Tracker(ConstantVelocity()))
        for i, (frame, _) in enumerate(simulate("single")):
            detections, measurements, tracks = tracker.analyze(frame)
            if i == 20:
                break
        self.assertTrue(tracks[0].forecasts)
        self.assertIsNotNone(tracks[0].direction)
        dashboard = render_preview(frame, detections, measurements, tracks, 20.0, SYNTHETIC_PROJECTION)
        self.assertEqual(dashboard.shape, (WINDOW_H, WINDOW_W, 3))
        waiting = render_preview(frame, [], [], [], 20.0, SYNTHETIC_PROJECTION)
        for i in range(4):
            left, top = panel_origin(i)
            region = np.s_[top:top + VIEW_H, left:left + PANEL_W]
            self.assertTrue(np.any(dashboard[region] != waiting[region]))
        left, top = panel_origin(3)
        overlay = np.s_[top:top + VIEW_H, left:left + PANEL_W]
        without_forecast = render_preview(frame, detections, measurements, [], 20.0, SYNTHETIC_PROJECTION)
        self.assertTrue(np.any(dashboard[overlay] != without_forecast[overlay]))

    def test_forecast_distinguishes_x_y_and_z(self):
        track = Estimate(1, "tennis_ball", (0, 0, 2), None, None, True, (),
                         (91, 67, 101, 77), (.01, .01, 1.99))
        cases = [
            ("X only", (.4, 0, 2), (96 + FOCAL * .2, 72), 1, 0, STEADY_DEPTH),
            ("Y only", (0, .4, 2), (96, 72 - FOCAL * .2), 1, 0, STEADY_DEPTH),
            ("Z closer", (0, 0, 1.5), (96, 72), 2 / 1.5, -.5, APPROACHING),
            ("Z farther", (0, 0, 2.5), (96, 72), 2 / 2.5, .5, RECEDING),
            ("X and Z", (.4, 0, 1.5), (96 + FOCAL * .4 / 1.5, 72), 2 / 1.5, -.5, APPROACHING),
        ]
        for name, target, center, scale, dz, color in cases:
            with self.subTest(name=name):
                forecast, = forecast_boxes(replace(track, forecasts=(target,)), SYNTHETIC_PROJECTION)
                self.assertEqual(forecast.horizon_s, .1)
                self.assertAlmostEqual(forecast.scale, scale)
                self.assertAlmostEqual(forecast.delta_z_m, dz)
                self.assertEqual(_depth_color(forecast.delta_z_m), color)
                x0, y0, x1, y1 = forecast.bounds
                np.testing.assert_allclose(((x0 + x1) / 2, (y0 + y1) / 2), center)
                self.assertAlmostEqual(x1 - x0, 10 * scale)

    def test_depth_readout_works_without_rgb_projection(self):
        track = Estimate(1, "tennis_ball", (0, 0, 2), None, None, True,
                         ((0, 0, 1.99), (0, 0, 1.75)))
        first, second = forecast_boxes(track, None)
        self.assertIsNone(first.bounds)
        self.assertIsNone(second.scale)
        self.assertEqual(_depth_color(first.delta_z_m), STEADY_DEPTH)
        self.assertEqual(second.delta_z_m, -.25)

    def test_projection_fits_sdk_style_samples(self):
        points = np.array([(x * z, y * z, z) for z in (1.0, 2.0, 3.5)
                           for x in (-.35, 0, .35) for y in (-.35, 0, .35)])
        uv = np.array([SYNTHETIC_PROJECTION.project(p)[:2] for p in points]).round()
        projection = ColorProjection.fit(points, uv)
        self.assertIsNotNone(projection)
        np.testing.assert_allclose(projection.project((.2, -.1, 1.4)),
                                   SYNTHETIC_PROJECTION.project((.2, -.1, 1.4)), atol=.5)
