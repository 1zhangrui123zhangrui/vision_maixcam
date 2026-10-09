from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import BoundingBox


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    input_width: int
    input_height: int


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        class_count: int,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if class_count <= 0:
            raise ValueError("class_count must be positive")
        self.class_count = class_count
        self.input_width, self.input_height = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.output_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR color image")
        blob_image, info = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            blob_image,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward(self.output_names)
        predictions = self._normalise_output(output[0])
        boxes: list[BoundingBox] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in predictions:
            if row.shape[0] == 4 + self.class_count:
                class_scores = row[4:]
            elif row.shape[0] == 5 + self.class_count:
                class_scores = row[5:] * row[4]
            else:
                raise ValueError(
                    f"unsupported YOLO output width {row.shape[0]} for {self.class_count} classes"
                )
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.confidence_threshold:
                continue
            x, y, w, h = map(float, row[:4])
            x1 = (x - w / 2.0 - info.pad_x) / info.scale
            y1 = (y - h / 2.0 - info.pad_y) / info.scale
            x2 = (x + w / 2.0 - info.pad_x) / info.scale
            y2 = (y + h / 2.0 - info.pad_y) / info.scale
            boxes.append(BoundingBox(x1, y1, x2, y2))
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []
        # Suppress repeated predictions within one class while preserving
        # different target labels if their boxes happen to overlap.
        indices: list[int] = []
        for class_id in sorted(set(class_ids)):
            class_indices = [i for i, value in enumerate(class_ids) if value == class_id]
            class_boxes = [[boxes[i].x1, boxes[i].y1, boxes[i].width, boxes[i].height] for i in class_indices]
            class_scores = [scores[i] for i in class_indices]
            keep = cv2.dnn.NMSBoxes(
                class_boxes,
                class_scores,
                self.confidence_threshold,
                self.nms_threshold,
            )
            if len(keep):
                indices.extend(class_indices[int(i)] for i in np.asarray(keep).reshape(-1).tolist())
        height, width = image.shape[:2]
        detections: list[RawDetection] = []
        for index in indices:
            box = boxes[index]
            clipped = BoundingBox(
                max(0.0, min(float(width), box.x1)),
                max(0.0, min(float(height), box.y1)),
                max(0.0, min(float(width), box.x2)),
                max(0.0, min(float(height), box.y2)),
            )
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            detections.append(RawDetection(class_ids[index], scores[index], clipped))
        return detections

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        height, width = image.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, LetterboxInfo(scale, pad_x, pad_y, self.input_width, self.input_height)

    def _normalise_output(self, output: np.ndarray) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"unsupported YOLO output shape: {output.shape}")
        expected_widths = {4 + self.class_count, 5 + self.class_count}
        if array.shape[1] in expected_widths:
            return array
        if array.shape[0] in expected_widths:
            return array.T
        raise ValueError(f"cannot identify YOLO output layout: {output.shape}")
