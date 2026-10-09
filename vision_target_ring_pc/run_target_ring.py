from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from target_ring import TargetRingRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run target-ring detection on one image")
    parser.add_argument("--model", required=True, type=Path, help="YOLO ONNX model path")
    parser.add_argument("--image", required=True, type=Path, help="input image path")
    parser.add_argument("--save", type=Path, help="optional annotated image path")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--nms", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")

    recognizer = TargetRingRecognizer(
        args.model,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
    )
    result, annotated = recognizer.detect(image, draw=True)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save), annotated):
            raise SystemExit(f"cannot write image: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
