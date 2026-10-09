"""Offline material-model smoke test; MaixCAM deployment does not import this file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from target_ring.yolo_onnx import YoloOnnxDetector


LABELS = ("black", "blue", "green", "light_blue", "red", "yellow")
COLOR_IDS = {"red": 1, "yellow": 2, "blue": 3, "green": 4, "black": 5, "light_blue": 6}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the material ONNX model on one image")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--nms", type=float, default=0.45)
    args = parser.parse_args()

    # imdecode handles Windows paths containing Chinese characters.
    image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    detector = YoloOnnxDetector(
        args.model,
        class_count=len(LABELS),
        # Exported ONNX graph is 640x640; training imgsz=1280 does not change
        # this deployment graph's fixed input size.
        input_size=(640, 640),
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
    )
    detections = detector.detect(image)
    result = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        label = LABELS[detection.class_id]
        result.append({
            "color": label,
            "color_id": COLOR_IDS[label],
            "confidence": round(detection.confidence, 4),
            "center": detection.bbox.center.to_dict(),
            "bbox": detection.bbox.to_dict(),
            "center_source": "top_circle_bbox",
        })
        box = detection.bbox
        p1 = (round(box.x1), round(box.y1))
        p2 = (round(box.x2), round(box.y2))
        center = (round(box.center.x), round(box.center.y))
        cv2.rectangle(image, p1, p2, (0, 220, 0), 2)
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(image, f"{label} {detection.confidence:.2f}", (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    print(json.dumps({"image_size": {"width": image.shape[1], "height": image.shape[0]}, "materials": result}, ensure_ascii=False, indent=2))
    if args.save and not cv2.imwrite(str(args.save), image):
        raise SystemExit(f"cannot write image: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
