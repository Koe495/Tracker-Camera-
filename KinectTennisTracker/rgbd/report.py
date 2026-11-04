import base64
import csv
import json
from dataclasses import asdict
from pathlib import Path
from struct import pack
from zlib import compress, crc32

from .frames import Frame, MAX_DEPTH_M, MIN_DEPTH_M
from .tracking import HORIZONS


def preview(frame: Frame, layer: str) -> str:
    raw = layer == "raw"
    width, height = (frame.depth_width, frame.depth_height) if raw else (frame.width, frame.height)
    stride = max(1, (width + 319) // 320)
    rows = bytearray()
    for y in range(0, height, stride):
        rows.append(0)  # PNG filter: none
        for x in range(0, width, stride):
            i = y * width + x
            if layer == "rgb":
                b, g, r = frame.bgrx[4 * i:4 * i + 3]
                rows.extend((r, g, b))
            else:
                z = frame.raw_depth_mm[i] / 1000 if raw else frame.color_xyz_m[3 * i + 2]
                if MIN_DEPTH_M <= z <= MAX_DEPTH_M:
                    value = round(255 * (MAX_DEPTH_M - z) / (MAX_DEPTH_M - MIN_DEPTH_M))
                    rows.extend((value, round(value * .8), 255 - value // 2))
                else:
                    rows.extend((15, 19, 28))

    def chunk(kind, data):
        return pack(">I", len(data)) + kind + data + pack(">I", crc32(kind + data))

    dimensions = ((width + stride - 1) // stride, (height + stride - 1) // stride)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", pack(">IIBBBBB", *dimensions, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def save_report(out: Path, rows, *, title: str, source: str, frame: Frame, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "title": title, "source": source, "metrics": metrics, "horizons": HORIZONS,
        "images": {layer: preview(frame, layer) for layer in ("rgb", "raw", "aligned")},
        "rows": [
            {"ms": ms - rows[0][0], "measurements": [m.xyz for m in measurements],
             "tracks": [asdict(track) for track in tracks]}
            for ms, measurements, tracks in rows
        ],
    }
    with (out / "tracks.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_s", "track_id", "label", "observed", "x_m", "y_m", "z_m",
            "vx_m_s", "vy_m_s", "vz_m_s", "dir_x", "dir_y", "dir_z",
            *(f"{axis}_m_at_{h:g}s" for h in HORIZONS for axis in "xyz"),
        ])
        for row in data["rows"]:
            for track in row["tracks"]:
                points = track["forecasts"] or [(None,) * 3] * len(HORIZONS)
                writer.writerow([
                    row["ms"] / 1000, track["id"], track["label"], int(track["observed"]),
                    *track["xyz"], *(track["velocity"] or [None] * 3), *(track["direction"] or [None] * 3),
                    *(coordinate for point in points for coordinate in point),
                ])
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    (out / "dashboard.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
