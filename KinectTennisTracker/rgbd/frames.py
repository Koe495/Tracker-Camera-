from dataclasses import dataclass
from struct import Struct
from typing import BinaryIO, Iterator

HEADER = Struct("<4s4H2q")
MIN_DEPTH_M, MAX_DEPTH_M = 0.8, 4.0
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    depth_width: int
    depth_height: int
    rgb_ms: int
    depth_ms: int
    bgrx: bytes
    raw_depth_mm: memoryview
    color_xyz_m: memoryview


def _read_exact(stream: BinaryIO, count: int, *, allow_eof: bool = False) -> bytes:
    data = stream.read(count)
    if len(data) == count or (not data and allow_eof):
        return data
    raise ValueError("Truncated RGB-D frame.")


def read_frames(stream: BinaryIO) -> Iterator[Frame]:
    previous_rgb = previous_depth = -1
    while header := _read_exact(stream, HEADER.size, allow_eof=True):
        magic, width, height, dw, dh, rgb_ms, depth_ms = HEADER.unpack(header)
        if magic != b"RGBD" or not all(0 < size <= 1920 for size in (width, height, dw, dh)):
            raise ValueError("Invalid RGB-D header.")
        if rgb_ms <= previous_rgb or depth_ms <= previous_depth or abs(rgb_ms - depth_ms) > 25:
            raise ValueError("Invalid RGB/depth timestamps.")
        yield Frame(
            width, height, dw, dh, rgb_ms, depth_ms,
            _read_exact(stream, width * height * 4),
            memoryview(_read_exact(stream, dw * dh * 2)).cast("H"),
            memoryview(_read_exact(stream, width * height * 12)).cast("f"),
        )
        previous_rgb, previous_depth = rgb_ms, depth_ms


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    stream.write(HEADER.pack(
        b"RGBD", frame.width, frame.height, frame.depth_width, frame.depth_height,
        frame.rgb_ms, frame.depth_ms,
    ))
    stream.write(frame.bgrx)
    stream.write(frame.raw_depth_mm)
    stream.write(frame.color_xyz_m)
