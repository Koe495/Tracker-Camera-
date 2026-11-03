"""Calibrated RGB/depth mapping through Kinect SDK 1.8 INuiCoordinateMapper.

ABI checked against the installed SDK's NuiSensor.h: sensor slot 28,
mapper slots 5 (color -> depth) and 8 (depth -> skeleton/camera XYZ).
"""
import ctypes as C

import numpy as np

from kinect_v1 import KinectError, KinectV1, check_hr


class SDKMapper:
    def __init__(self, dll):
        self.dll = dll
        self.sensor, self.mapper = C.c_void_p(), C.c_void_p()
        self.pixels = np.zeros((480, 640, 2), np.uint16)  # playerIndex, depth mm
        self.points = np.empty((480, 640, 4), np.int32)  # x,y,depth,reserved
        self.xyz4 = np.empty((480, 640, 4), np.float32)

    def __enter__(self):
        self.dll.NuiCreateSensorByIndex.argtypes = [C.c_int32, C.POINTER(C.c_void_p)]
        self.dll.NuiCreateSensorByIndex.restype = C.c_int32
        check_hr(self.dll.NuiCreateSensorByIndex(0, C.byref(self.sensor)), "Get Kinect calibration sensor")
        try:
            get_mapper = KinectV1._method(self.sensor, 28, C.c_int32, C.POINTER(C.c_void_p))
            check_hr(get_mapper(self.sensor, C.byref(self.mapper)), "Get coordinate mapper")
            if not self.mapper:
                raise KinectError("SDK returned no coordinate mapper.")
            self.map_color = KinectV1._method(self.mapper, 5, C.c_int32, C.c_int32,
                C.c_int32, C.c_int32, C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p)
            self.map_xyz = KinectV1._method(self.mapper, 8, C.c_int32, C.c_int32,
                C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p)
            return self
        except Exception:
            self.__exit__()
            raise

    def align(self, native_depth):
        if native_depth.shape != (480, 640) or native_depth.dtype != np.uint16:
            raise ValueError("SDK mapping requires native 640x480 uint16 millimetres.")
        self.pixels[:, :, 1] = native_depth
        check_hr(self.map_color(self.mapper, 1, 2, 2, native_depth.size,
                 self.pixels.ctypes.data, native_depth.size, self.points.ctypes.data),
                 "Map RGB to depth")
        check_hr(self.map_xyz(self.mapper, 2, native_depth.size, self.pixels.ctypes.data,
                 native_depth.size, self.xyz4.ctypes.data), "Map depth to camera XYZ")
        x, y = self.points[:, :, 0], self.points[:, :, 1]
        valid = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
        aligned = np.zeros((480, 640), np.uint16)
        aligned[valid] = native_depth[y[valid], x[valid]]
        # Keep holes as zero: no interpolation across object boundaries.
        return aligned, self.xyz4[:, :, :3].copy()

    def __exit__(self, *_):
        for pointer in (self.mapper, self.sensor):
            if pointer.value:
                KinectV1._method(pointer, 2, C.c_uint32)(pointer)
                pointer.value = None


class DemoMapper:
    """Identity mapping only for explicitly synthetic, already aligned images."""
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def align(self, native_depth):
        y, x = np.mgrid[:480, :640]
        z = native_depth.astype(np.float32) / 1000
        xyz = np.stack(((x - 319.5) * z / 571.26,
                        (239.5 - y) * z / 571.26, z), axis=-1).astype(np.float32)
        xyz[z == 0] = np.nan
        return native_depth.copy(), xyz
