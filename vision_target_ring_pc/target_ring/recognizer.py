from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .center import estimate_center
from .types import BoundingBox, TargetDetection, TargetRingResult
from .yolo_onnx import YoloOnnxDetector


LABELS = ("target_1", "target_2", "target_3")


class TargetRingRecognizer:
    """Independent recognizer for the three target rings."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
    ) -> None:
        self.detector = YoloOnnxDetector(
            model_path,
            class_count=len(LABELS),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        draw: bool = False,
    ) -> TargetRingResult | tuple[TargetRingResult, np.ndarray]:
        detections = self.detector.detect(image)
        targets: dict[str, TargetDetection | None] = {label: None for label in LABELS}
        for raw in detections:
            label = LABELS[raw.class_id]
            # NMS is class-aware in practice here because each target label is
            # spatially distinct. Keep the strongest result if a class repeats.
            current = targets[label]
            if current is not None and current.confidence >= raw.confidence:
                continue
            center = estimate_center(image, raw.bbox)
            targets[label] = TargetDetection(
                label=label,
                class_id=raw.class_id,
                confidence=raw.confidence,
                bbox=raw.bbox,
                bbox_center=raw.bbox.center,
                center=center.point,
                center_source=center.source,
                touches_image_edge=_touches_edge(raw.bbox, image.shape[1], image.shape[0]),
                ring_count=center.ring_count,
                center_quality=center.quality,
                center_usable=center.usable,
            )

        result = TargetRingResult(image.shape[1], image.shape[0], targets)
        if not draw:
            return result
        return result, self._draw(image, result)

    def _draw(self, image: np.ndarray, result: TargetRingResult) -> np.ndarray:
        annotated = image.copy()
        for label, detection in result.targets.items():
            if detection is None:
                continue
            box = detection.bbox
            x1, y1, x2, y2 = map(lambda value: int(round(value)), (box.x1, box.y1, box.x2, box.y2))
            cx, cy = int(round(detection.center.x)), int(round(detection.center.y))
            color = ((0, 255, 0), (0, 180, 255), (255, 100, 0))[detection.class_id]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(annotated, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            text = f"{label} {detection.confidence:.2f} ({cx},{cy}) {detection.center_source} n={detection.ring_count}"
            cv2.putText(annotated, text, (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated


def _touches_edge(bbox: BoundingBox, width: int, height: int, margin: int = 2) -> bool:
    return bbox.x1 <= margin or bbox.y1 <= margin or bbox.x2 >= width - margin or bbox.y2 >= height - margin
