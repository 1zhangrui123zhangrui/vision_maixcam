from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from target_ring import TargetRingRecognizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()
    recognizer = TargetRingRecognizer(args.model)
    paths = sorted(args.images.glob("*.jpg")) if args.images.is_dir() else [args.images]
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip unreadable: {path}")
            continue
        result, annotated = recognizer.detect(image, draw=True)
        print(path.name)
        for label, item in result.targets.items():
            if item is not None:
                print(f"  {label}: center=({item.center.x:.1f},{item.center.y:.1f}) source={item.center_source} rings={item.ring_count} quality={item.center_quality:.3f} usable={item.center_usable}")
        if args.save_dir:
            cv2.imwrite(str(args.save_dir / path.name), annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
