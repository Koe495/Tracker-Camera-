"""Hardware-independent checks of the SDK ABI and metric depth pipeline."""
import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kinect_v1 import Frame, ImageFrame, KinectError, KinectV1, LockedRect, copy_locked_rect, unpack_depth
from main import parse_args, save_pair
from vision import Pairer, dashboard, depth_image


class VisionTests(unittest.TestCase):
    def test_sdk_struct_layout(self):
        self.assertEqual(C.sizeof(ImageFrame), 48 if C.sizeof(C.c_void_p) == 8 else 40)
        self.assertEqual(ImageFrame.texture.offset, 24 if C.sizeof(C.c_void_p) == 8 else 20)
        self.assertEqual(C.sizeof(LockedRect), 16 if C.sizeof(C.c_void_p) == 8 else 12)

    def test_packed_depth_is_shifted_and_sentinels_are_removed(self):
        packed = np.array([0, (1000 << 3) | 7, 4000 << 3, 4095 << 3, 8191 << 3], np.uint16)
        np.testing.assert_array_equal(unpack_depth(packed), [0, 1000, 4000, 0, 0])

    def test_padded_sdk_rows_and_buffer_ownership(self):
        packed = np.array([[800 << 3, 1200 << 3, 999], [2400 << 3, 3500 << 3, 999]], np.uint16)
        rect = LockedRect(6, packed.nbytes, packed.ctypes.data)
        result = copy_locked_rect(rect, 2, 2, False)
        packed[:] = 0  # data must survive SDK frame release
        np.testing.assert_array_equal(result, [[800, 1200], [2400, 3500]])

    def test_bgrx_not_rgb_and_padding_is_ignored(self):
        raw = np.array([[10, 20, 30, 255, 99, 99, 99, 99]], np.uint8)
        result = copy_locked_rect(LockedRect(8, 8, raw.ctypes.data), 1, 1, True)
        np.testing.assert_array_equal(result, [[[10, 20, 30]]])

    def test_invalid_buffer_is_rejected(self):
        with self.assertRaises(KinectError):
            copy_locked_rect(LockedRect(0, 0, None), 640, 480, False)

    def test_pairing_rejects_stale_frames_and_never_reuses_frames(self):
        frame = lambda t: Frame(np.zeros((1, 1), np.uint16), t, t)
        pairer = Pairer(20)
        self.assertIsNone(pairer.add(frame(10), frame(100)))
        pair = pairer.add(frame(101), None)
        self.assertEqual([f.timestamp_ms for f in pair], [101, 100])
        self.assertIsNone(pairer.add(None, None))

    def test_native_frame_is_released_even_if_texture_copy_fails(self):
        native = ImageFrame()
        native.texture = 123
        native.timestamp = 500
        frame_pointer = C.pointer(native)

        class FakeDLL:
            released = False

            def NuiImageStreamGetNextFrame(self, handle, timeout, out):
                C.cast(out, C.POINTER(C.POINTER(ImageFrame)))[0] = frame_pointer
                return 0

            def NuiImageStreamReleaseFrame(self, handle, frame):
                self.released = True
                return 0

        sensor = KinectV1()
        sensor.dll = FakeDLL()
        sensor.streams = {"depth": C.c_void_p(1)}
        actions = []

        def method(texture, index, *args):
            def call(*args):
                actions.append(index)
                return 0
            return call

        with patch.object(sensor, "_method", side_effect=method):
            with self.assertRaises(KinectError):
                sensor.read("depth")  # Lock succeeds but buffer is empty
        self.assertEqual(actions, [5, 7])
        self.assertTrue(sensor.dll.released)

    def test_invalid_depth_is_black_and_metric_scale_is_fixed(self):
        depth = np.array([[0, 700, 800, 2000, 4000, 4095]], np.uint16)
        image = depth_image(depth, 800, 4000)
        np.testing.assert_array_equal(image[0, [0, 1, 5]], np.zeros((3, 3), np.uint8))
        same_depth = depth_image(np.array([[2000]], np.uint16), 800, 4000)
        np.testing.assert_array_equal(image[0, 3], same_depth[0, 0])

    def test_snapshot_preserves_uint16_metric_depth_and_metadata(self):
        rgb = Frame(np.full((480, 640, 3), 120, np.uint8), 1000, 30)
        depth = Frame(np.full((480, 640), 2345, np.uint16), 1003, 30)
        with tempfile.TemporaryDirectory() as directory:
            folder = save_pair(Path(directory), rgb, depth, 800, 4000, True)
            saved = cv2.imdecode(np.fromfile(folder / "depth_mm.png", np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertEqual(saved.dtype, np.uint16)
            np.testing.assert_array_equal(saved, depth.data)
            meta = json.loads((folder / "metadata.json").read_text())
            self.assertEqual(meta["depth_unit"], "millimetres")
            self.assertEqual(meta["source"], "simulation")
        view = dashboard(rgb, depth, 800, 4000, 30, True)
        self.assertEqual(view.shape, (646, 1308, 3))

    def test_invalid_cli_ranges_fail_early(self):
        with self.assertRaises(SystemExit):
            parse_args(["--min-depth", "4000", "--max-depth", "800"])


if __name__ == "__main__":
    unittest.main()
