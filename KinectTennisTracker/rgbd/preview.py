"""Draw the four synchronized views for the live Kinect window."""

import cv2
import numpy as np

from .frames import MAX_DEPTH_M, MIN_DEPTH_M
from .projection import forecast_boxes

PANEL_W, VIEW_H = 368, 276
GAP, VIEW_TOP = 12, 72
PANEL_TOP, PANEL_H = 36, VIEW_H + 93
ROW_PITCH = PANEL_H + GAP
WINDOW_W, WINDOW_H = 2 * PANEL_W + 3 * GAP, PANEL_TOP + ROW_PITCH + PANEL_H + 27
BACKGROUND = (25, 21, 18)
SURFACE = (40, 34, 29)
TEXT = (245, 239, 230)
MUTED = (194, 181, 168)
GRID = (76, 67, 57)
MEASURED = (145, 229, 113)
MISSING = (76, 188, 255)
TRACK_COLORS = (MEASURED, (120, 195, 254), (248, 167, 175))
APPROACHING = (85, 177, 255)
RECEDING = (246, 224, 89)
STEADY_DEPTH = (218, 216, 211)
DEPTH_COLOR_THRESHOLD_M = .03
FULL_OPACITY_Z_SPEED_M_S = 2.0
RANGE_BAR = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[::-1].copy()[None, :], cv2.COLORMAP_TURBO)


def _text(image, value, position, *, scale=.43, color=TEXT):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def colorize_depth(depth):
    valid = (MIN_DEPTH_M <= depth) & (depth <= MAX_DEPTH_M)
    shade = np.zeros(depth.shape, dtype=np.uint8)
    shade[valid] = np.rint((MAX_DEPTH_M - depth[valid]) * 255 / (MAX_DEPTH_M - MIN_DEPTH_M))
    color = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
    color[~valid] = BACKGROUND
    return color


def panel_origin(index):
    return GAP + (index % 2) * (PANEL_W + GAP), VIEW_TOP + (index // 2) * ROW_PITCH


def _view_box(box, frame):
    x0, y0, x1, y1 = box
    return (round(x0 * PANEL_W / frame.width), round(y0 * VIEW_H / frame.height),
            round(x1 * PANEL_W / frame.width), round(y1 * VIEW_H / frame.height))


def _depth_color(delta_z_m):
    if delta_z_m < -DEPTH_COLOR_THRESHOLD_M:
        return APPROACHING
    if delta_z_m > DEPTH_COLOR_THRESHOLD_M:
        return RECEDING
    return STEADY_DEPTH


def _depth_opacity(delta_z_m, horizon_s):
    speed_fraction = min(1.0, abs(delta_z_m) / (horizon_s * FULL_OPACITY_Z_SPEED_M_S))
    return .35 + .60 * speed_fraction


def _draw_dashed_box(view, box, color, opacity):
    x0, y0, x1, y1 = box
    left, top = max(0, x0 - 2), max(0, y0 - 2)
    region = view[top:y1 + 3, left:x1 + 3]
    layer = region.copy()
    corners = ((x0 - left, y0 - top), (x1 - left, y0 - top),
               (x1 - left, y1 - top), (x0 - left, y1 - top))
    for start, end in zip(corners, (*corners[1:], corners[0])):
        visible, start, end = cv2.clipLine((0, 0, layer.shape[1], layer.shape[0]), start, end)
        if not visible:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = abs(dx) + abs(dy)
        for offset in range(0, length, 8):
            stop = min(offset + 4, length)
            a = (start[0] + round(dx * offset / length), start[1] + round(dy * offset / length))
            b = (start[0] + round(dx * stop / length), start[1] + round(dy * stop / length))
            cv2.line(layer, a, b, color, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, opacity, region, 1 - opacity, 0, dst=region)


def _draw_ghosts(view, frame, predictions):
    for track, steps in predictions:
        current_box = _view_box(track.box, frame) if track.observed and track.box is not None else None
        separated = []
        for step in reversed(steps):
            if step.bounds is None:
                continue
            box = _view_box(step.bounds, frame)
            if box[2] <= 0 or box[0] >= PANEL_W or box[3] <= 0 or box[1] >= VIEW_H:
                continue
            if current_box is not None and max(abs(a - b) for a, b in zip(box, current_box)) < 2:
                continue
            if separated and max(abs(a - b) for a, b in zip(box, separated[-1][0])) < 2:
                continue
            separated.append((box, step))
        separated.reverse()
        for index in range(len(separated) - 1, -1, -1):
            box, step = separated[index]
            x0, _, _, y1 = box
            color = _depth_color(step.delta_z_m)
            _draw_dashed_box(view, box, color, _depth_opacity(step.delta_z_m, step.horizon_s))
            far_box = separated[-1][0]
            if index == len(separated) - 1 or (index == 0 and (abs(far_box[0] - x0) > 48 or abs(far_box[3] - y1) > 18)):
                lx = min(max(x0, 4), PANEL_W - 72)
                ly = min(max(y1 + 12, 12), VIEW_H - 4)
                label = f"+{step.horizon_s:.1f}s"
                if len(predictions) > 1:
                    label = f"#{track.id} {label}"
                width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .37, 1)[0][0]
                cv2.rectangle(view, (lx - 2, ly - 10), (lx + width + 2, ly + 3), BACKGROUND, -1)
                _text(view, label, (lx, ly), scale=.37, color=color)


def _camera_views(frame, detections, measurements, predictions, rgb_view, depth_view, forecast_view):
    rgb = np.frombuffer(frame.bgrx, dtype=np.uint8).reshape(frame.height, frame.width, 4)
    depth = np.asarray(frame.color_xyz_m).reshape(frame.height, frame.width, 3)[:, :, 2]
    rgb_view[:] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR), (PANEL_W, VIEW_H),
                             interpolation=cv2.INTER_AREA)
    sampled = cv2.resize(depth, (PANEL_W, VIEW_H), interpolation=cv2.INTER_NEAREST)
    depth_view[:] = colorize_depth(sampled)
    forecast_view[:] = rgb_view
    _draw_ghosts(forecast_view, frame, predictions)

    xyz_by_box = {item.detection.bbox: item.xyz for item in measurements}
    for detection in detections:
        x0, y0, x1, y1 = _view_box(detection.bbox, frame)
        xyz = xyz_by_box.get(detection.bbox)
        color = MEASURED if xyz is not None else MISSING
        for view in (rgb_view, depth_view, forecast_view):
            cv2.rectangle(view, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 1, cv2.LINE_AA)
        for view, value in (
            (rgb_view, f"Tennis {detection.confidence:.2f}"),
            (depth_view, f"Z {xyz[2]:.2f} m" if xyz is not None else "No valid depth"),
        ):
            label_w = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0][0] + 8
            lx, ly = min(x0, PANEL_W - label_w), max(y0 - 16, 0)
            cv2.rectangle(view, (lx, ly), (lx + label_w, ly + 15), BACKGROUND, -1)
            _text(view, value, (lx + 4, ly + 11), scale=.42, color=color)


def _depth_forecast_caption(canvas, x, top, predictions, projection):
    if predictions:
        # Keep the readout on the lowest tracked ID with a current measurement.
        track, steps = min(predictions, key=lambda item: (not item[0].observed, item[0].id))
        label = f"#{track.id}  dZ (m) vs now" + ("" if track.observed else "  /  coasting")
        _text(canvas, label, (x, top + 16))
        last = steps[-1]
        if last.scale is not None:
            _text(canvas, f"Box +{last.horizon_s:.1f}s: x{last.scale:.2f}", (x + 204, top + 16), scale=.40)
        elif projection is None:
            _text(canvas, "RGB map unavailable", (x + 204, top + 16), scale=.40, color=MUTED)
        for i, step in enumerate(steps):
            label = f"{step.horizon_s:.1f}s {step.delta_z_m:+.2f}"
            _text(canvas, label, (x + 68 * i, top + 34), scale=.39, color=_depth_color(step.delta_z_m))
    else:
        status = "RGB projection unavailable" if projection is None else "Waiting for a 3D forecast"
        _text(canvas, status, (x, top + 23), color=MUTED)
    _text(canvas, "Closer", (x, top + 51), scale=.39, color=APPROACHING)
    _text(canvas, "Farther", (x + 83, top + 51), scale=.39, color=RECEDING)
    _text(canvas, "Similar depth (within 3 cm)", (x + 172, top + 51), scale=.39, color=STEADY_DEPTH)


def _prediction_view(plot, tracks):
    plot[:] = BACKGROUND

    def project(point):
        x, y, z = point
        return round(172 + 49 * x + 20 * z), round(234 + 8 * x - 57 * y - 29 * z)

    def line(start, end, color=GRID):
        cv2.line(plot, project(start), project(end), color, 1, cv2.LINE_AA)

    for x in range(-2, 3):
        line((x, 0, 0), (x, 0, 4))
    for z in range(5):
        line((-2, 0, z), (2, 0, z))

    origin = project((0, 0, 0))
    for axis, point in (("X", (2.2, 0, 0)), ("Y", (0, 2, 0)), ("Z", (0, 0, 4.35))):
        end = project(point)
        cv2.arrowedLine(plot, origin, end, MUTED, 1, cv2.LINE_AA, tipLength=.07)
        _text(plot, axis, (end[0] + 5, end[1] - 4), color=MUTED)
    _text(plot, "Kinect", (origin[0] - 47, origin[1] + 21), color=MUTED)

    for track in tracks:
        color = TRACK_COLORS[(track.id - 1) % len(TRACK_COLORS)]
        point = project(track.xyz)
        if track.direction is not None:
            tip = tuple(p + .65 * d for p, d in zip(track.xyz, track.direction))
            cv2.arrowedLine(plot, point, project(tip), color, 2, cv2.LINE_AA, tipLength=.25)
        previous = track.xyz
        for future in track.forecasts:
            line(previous, future, color)
            cv2.circle(plot, project(future), 3, color, 1, cv2.LINE_AA)
            previous = future
        cv2.circle(plot, point, 6, color, -1 if track.observed else 2, cv2.LINE_AA)
        _text(plot, f"#{track.id}", (point[0] + 8, point[1] - 8), color=color)


def render_preview(frame, detections, measurements, tracks, fps, projection=None):
    canvas = np.full((WINDOW_H, WINDOW_W, 3), BACKGROUND, dtype=np.uint8)
    _text(canvas, "Kinect Tennis Tracker", (GAP, 23), scale=.60)
    _text(canvas, f"{fps:.1f} FPS  |  RGB-depth skew {abs(frame.rgb_ms - frame.depth_ms)} ms",
          (WINDOW_W - 282, 23), color=MUTED)

    views = []
    for i, title in enumerate(("01  RGB / BALL DETECTION", "02  DEPTH / RGB ALIGNED",
                               "03  3D MOTION PREDICTION", "04  RGB / FORECAST BOXES")):
        left, top = panel_origin(i)
        row = PANEL_TOP + i // 2 * ROW_PITCH
        panel = canvas[row:row + PANEL_H, left:left + PANEL_W]
        panel[:] = SURFACE
        _text(canvas, title, (left + GAP, top - 13), scale=.49)
        views.append(canvas[top:top + VIEW_H, left:left + PANEL_W])

    predictions = [(track, steps) for track in tracks if (steps := tuple(forecast_boxes(track, projection)))]
    _camera_views(frame, detections, measurements, predictions, views[0], views[1], views[3])
    _prediction_view(views[2], tracks)

    (rgb_left, _), (depth_left, _), (plot_left, plot_top), (forecast_left, forecast_top) = map(panel_origin, range(4))
    bottom = VIEW_TOP + VIEW_H
    _text(canvas, f"{len(detections)} RGB candidate(s)  /  {len(measurements)} with valid XYZ",
          (rgb_left + GAP, bottom + 23))
    _text(canvas, "Green: valid XYZ   |   Amber: RGB only", (rgb_left + GAP, bottom + 42), color=MUTED)
    bar_x = depth_left + GAP
    canvas[bottom + 12:bottom + 22, bar_x:bar_x + PANEL_W - 2 * GAP] = cv2.resize(RANGE_BAR, (PANEL_W - 2 * GAP, 10))
    _text(canvas, f"{MIN_DEPTH_M:.1f} m / near", (bar_x, bottom + 42), color=MUTED)
    _text(canvas, "Dark: invalid", (bar_x + 133, bottom + 42), color=MUTED)
    _text(canvas, f"{MAX_DEPTH_M:.1f} m / far", (bar_x + 277, bottom + 42), color=MUTED)
    if not tracks:
        _text(canvas, "Waiting for a ball with valid depth", (plot_left + GAP, plot_top + VIEW_H + 26), color=MUTED)
    for i, track in enumerate(tracks):
        xyz = "  ".join(f"{value:+.2f}" for value in track.xyz)
        state = ("coasting" if not track.observed else
                 "measured" if track.forecasts else "collecting measurements")
        _text(canvas, f"#{track.id}  XYZ {xyz} m  /  {state}", (plot_left + GAP, plot_top + VIEW_H + 16 + 15 * i),
              color=TRACK_COLORS[(track.id - 1) % len(TRACK_COLORS)])
    _depth_forecast_caption(canvas, forecast_left + GAP, forecast_top + VIEW_H, predictions, projection)
    _text(canvas, "Q / Esc: quit  |  3D arrow: direction", (GAP, WINDOW_H - 9), color=MUTED)
    _text(canvas, "Dashed: forecast  |  Bolder: faster along Z", (forecast_left + GAP, WINDOW_H - 9), color=MUTED)
    return canvas
