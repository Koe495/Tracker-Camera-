"""Single-class YOLOv8 ONNX detector, returning boxes in Kinect color pixels."""

from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .detection import Detection2D

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "tennis-ball.onnx"


def center_mask(bbox, width):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = .34 * (x1 - x0), .34 * (y1 - y0)
    return tuple(
        y * width + x
        for y in range(y0, y1)
        for x in range(x0, x1)
        if ((x + .5 - cx) / rx)**2 + ((y + .5 - cy) / ry)**2 <= 1
    )


class TennisDetector:
    def __init__(self, weights=DEFAULT_MODEL, confidence=.25, threads=4, image_size=640):
        if not 0 < confidence <= 1 or threads < 1:
            raise ValueError("Confidence must be in (0, 1] and threads must be positive.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        batch, channels, height, width = self.input.shape
        if (isinstance(batch, int) and batch != 1) or channels != 3:
            raise ValueError("Expected a YOLOv8 ONNX image input with three channels and batch size 1.")
        self.width = width if isinstance(width, int) else image_size
        self.height = height if isinstance(height, int) else image_size
        self.confidence = confidence

    def detect(self, bgrx: bytes, width: int, height: int) -> list[Detection2D]:
        image = np.frombuffer(bgrx, dtype=np.uint8).reshape(height, width, 4)
        return self.detect_image(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))

    def detect_image(self, bgr: np.ndarray) -> list[Detection2D]:
        height, width = bgr.shape[:2]
        scale = min(self.width / width, self.height / height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        xpad, ypad = (self.width - w) // 2, (self.height - h) // 2
        image = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        image[ypad:ypad + h, xpad:xpad + w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255
        prediction = self.session.run(None, {self.input.name: tensor})[0][0]
        if prediction.ndim != 2 or 5 not in prediction.shape:
            raise ValueError(f"Expected single-class YOLOv8 predictions, received {prediction.shape}.")
        if prediction.shape[0] == 5:
            prediction = prediction.T
        candidates = prediction[prediction[:, 4] >= self.confidence]
        boxes, scores = [], []
        for cx, cy, bw, bh, score in candidates:
            x0 = max(0, floor((cx - bw / 2 - xpad) * width / w))
            y0 = max(0, floor((cy - bh / 2 - ypad) * height / h))
            x1 = min(width, ceil((cx + bw / 2 - xpad) * width / w))
            y1 = min(height, ceil((cy + bh / 2 - ypad) * height / h))
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                boxes.append([x0, y0, x1 - x0, y1 - y0])
                scores.append(float(score))
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, max(0.0, self.confidence - 1e-7), .45)
        detections = []
        for i in np.asarray(indices).reshape(-1):
            x, y, w, h = boxes[i]
            bbox = (x, y, x + w, y + h)
            detections.append(Detection2D(bbox, scores[i], "tennis_ball", center_mask(bbox, width)))
        return detections
