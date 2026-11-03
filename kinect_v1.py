"""Minimal Windows Kinect SDK 1.8 binding for Kinect for Windows model 1517.

Uses the exported (global) C API, whose GetNextFrame returns a frame POINTER.
This differs from the INuiSensor COM method that fills a caller-owned struct.
No obsolete Python 2 PyKinect wrapper is required.
"""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class KinectError(RuntimeError):
    pass


class ViewArea(C.Structure):
    _fields_ = [("zoom", C.c_int32), ("center_x", C.c_int32), ("center_y", C.c_int32)]


class ImageFrame(C.Structure):
    _fields_ = [
        ("timestamp", C.c_int64), ("number", C.c_uint32),
        ("image_type", C.c_int32), ("resolution", C.c_int32),
        ("texture", C.c_void_p), ("flags", C.c_uint32), ("view", ViewArea),
    ]


class LockedRect(C.Structure):
    _fields_ = [("pitch", C.c_int32), ("size", C.c_int32), ("bits", C.c_void_p)]


@dataclass
class Frame:
    data: np.ndarray
    timestamp_ms: int
    number: int


def unpack_depth(packed: np.ndarray) -> np.ndarray:
    """NUI packed ushort: bits 3..15 hold millimetres; 0/4095/8191 are invalid."""
    depth = np.right_shift(packed, 3).astype(np.uint16)
    depth[(depth == 4095) | (depth == 8191)] = 0
    return depth


def copy_locked_rect(rect: LockedRect, width: int, height: int, color: bool) -> np.ndarray:
    """Copy while the SDK texture is locked, respecting padded row stride."""
    row_bytes = width * (4 if color else 2)
    if not rect.bits or rect.pitch < row_bytes or rect.size < rect.pitch * height:
        raise KinectError("Invalid Kinect texture buffer or row stride.")
    buffer = (C.c_uint8 * (rect.pitch * height)).from_address(rect.bits)
    rows = np.ctypeslib.as_array(buffer).reshape(height, rect.pitch)
    raw = rows[:, :row_bytes].copy()
    if color:
        return raw.reshape(height, width, 4)[:, :, :3].copy()  # SDK BGRX -> OpenCV BGR
    return unpack_depth(raw.view(np.uint16).reshape(height, width))


def check_hr(result: int, operation: str) -> None:
    if result < 0:
        raise KinectError(
            f"{operation} failed (HRESULT 0x{result & 0xFFFFFFFF:08X}). "
            "Check external power, USB connection and Kinect SDK 1.8; "
            "close Kinect Studio and other camera applications."
        )


def load_runtime():
    if os.name != "nt":
        raise KinectError("Kinect model 1517 requires Windows and Kinect SDK 1.8.")
    # Explicit system path avoids loading a DLL from an untrusted working directory.
    runtime = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "Kinect10.dll"
    if not runtime.is_file():
        raise KinectError(
            "Missing Kinect10.dll. Install Microsoft Kinect for Windows SDK 1.8 "
            "(model 1517 uses v1, not SDK 2.0).\n"
            "https://www.microsoft.com/en-us/download/details.aspx?id=40278\n"
            "Use --demo to preview without a camera."
        )
    try:
        return C.WinDLL(str(runtime))
    except OSError as exc:
        raise KinectError(f"Cannot load Kinect SDK 1.8 runtime: {exc}") from exc


class KinectV1:
    width, height = 640, 480

    def __init__(self, near_mode: bool = False):
        self.near_mode = near_mode
        self.dll = None
        self.initialized = False
        self.streams: dict[str, C.c_void_p] = {}

    def open(self):
        self.dll = load_runtime()
        dll = self.dll
        dll.NuiInitialize.argtypes = [C.c_uint32]
        dll.NuiInitialize.restype = C.c_int32
        dll.NuiShutdown.argtypes = []
        dll.NuiShutdown.restype = None
        dll.NuiImageStreamOpen.argtypes = [C.c_int32, C.c_int32, C.c_uint32,
                                           C.c_uint32, C.c_void_p, C.POINTER(C.c_void_p)]
        dll.NuiImageStreamOpen.restype = C.c_int32
        dll.NuiImageStreamGetNextFrame.argtypes = [C.c_void_p, C.c_uint32,
                                                  C.POINTER(C.POINTER(ImageFrame))]
        dll.NuiImageStreamGetNextFrame.restype = C.c_int32
        dll.NuiImageStreamReleaseFrame.argtypes = [C.c_void_p, C.POINTER(ImageFrame)]
        dll.NuiImageStreamReleaseFrame.restype = C.c_int32
        check_hr(dll.NuiInitialize(0x02 | 0x20), "NuiInitialize")  # COLOR | DEPTH
        self.initialized = True
        try:
            for name, kind in (("color", 1), ("depth", 4)):
                handle = C.c_void_p()
                flags = 0x10000  # S_FALSE when no frame; poll without blocking UI
                if name == "depth":
                    flags |= 0x40000  # distinct too-near / too-far sentinels
                    if self.near_mode:
                        flags |= 0x20000
                check_hr(dll.NuiImageStreamOpen(kind, 2, flags, 2, None, C.byref(handle)),
                         f"Open {name} stream")
                self.streams[name] = handle
        except Exception:
            self.close()
            raise
        return self

    @staticmethod
    def _method(texture, index, result_type, *arguments):
        table = C.cast(texture, C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(result_type, C.c_void_p, *arguments)(table[index])

    def read(self, stream: str) -> Frame | None:
        pointer = C.POINTER(ImageFrame)()
        handle = self.streams[stream]
        hr = self.dll.NuiImageStreamGetNextFrame(handle, 0, C.byref(pointer))
        if hr == 1 or (hr & 0xFFFFFFFF) == 0x83010001:  # S_FALSE / E_NUI_FRAME_NO_DATA
            return None
        check_hr(hr, f"Read {stream}")
        if not pointer:
            raise KinectError(f"SDK returned an empty {stream} frame.")
        try:
            frame = pointer.contents
            if not frame.texture:
                raise KinectError("SDK returned an empty frame texture.")
            lock = self._method(frame.texture, 5, C.c_int32, C.c_uint32,
                                C.POINTER(LockedRect), C.c_void_p, C.c_uint32)
            unlock = self._method(frame.texture, 7, C.c_int32, C.c_uint32)
            rect = LockedRect()
            check_hr(lock(frame.texture, 0, C.byref(rect), None, 0), "Lock texture")
            try:
                data = copy_locked_rect(rect, self.width, self.height, stream == "color")
            finally:
                check_hr(unlock(frame.texture, 0), "Unlock texture")
            return Frame(data, int(frame.timestamp), int(frame.number))
        finally:
            check_hr(self.dll.NuiImageStreamReleaseFrame(handle, pointer), "Release frame")

    def close(self):
        if self.initialized:
            self.dll.NuiShutdown()
            self.initialized = False
            self.streams.clear()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


def diagnose() -> str:
    dll = load_runtime()
    dll.NuiGetSensorCount.argtypes = [C.POINTER(C.c_int32)]
    dll.NuiGetSensorCount.restype = C.c_int32
    count = C.c_int32()
    check_hr(dll.NuiGetSensorCount(C.byref(count)), "NuiGetSensorCount")
    return f"Kinect SDK 1.8 runtime OK; Python {C.sizeof(C.c_void_p) * 8}-bit; sensors: {count.value}"
