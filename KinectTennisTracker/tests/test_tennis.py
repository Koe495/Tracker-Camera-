import unittest

import numpy as np

from rgbd.tennis import TennisDetector, center_mask


class TennisDetectorTests(unittest.TestCase):
    def test_maps_letterbox_and_suppresses_duplicate(self):
        class MockSession:
            def run(self, outputs, inputs):
                tensor = inputs["images"]
                assert tensor.shape == (1, 3, 640, 640)
                assert np.isclose(tensor[0, 0, 0, 0], 114 / 255)
                assert np.allclose(tensor[0, :, 81, 0], [1, 0, 0])
                return [np.array([[
                    [320, 321, 80], [240, 241, 100], [32, 32, 16], [32, 32, 16], [.8, .7, .1],
                ]], dtype=np.float32)]

        detector = TennisDetector.__new__(TennisDetector)
        detector.session = MockSession()
        detector.input = type("Input", (), {"name": "images"})()
        detector.width = detector.height = 640
        detector.confidence = .25
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        detection, = detector.detect_image(bgr)
        self.assertEqual(detection.bbox, (304, 144, 336, 176))
        self.assertEqual(detection.label, "tennis_ball")
        self.assertIn(160 * 640 + 320, detection.mask_pixels)

    def test_depth_mask_stays_in_ball_center(self):
        points = center_mask((10, 20, 20, 30), 64)
        self.assertGreaterEqual(len(points), 8)
        self.assertNotIn(20 * 64 + 10, points)
        self.assertIn(25 * 64 + 15, points)
