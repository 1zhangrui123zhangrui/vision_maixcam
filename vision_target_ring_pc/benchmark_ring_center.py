"""PC estimator-only timing; real photographs have no ground-truth labels."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision_target_ring_maixcam"))
from ring_center import refine_center
import config
from test_ring_center import sample, BOX


def measure(rgb, box, max_side):
    for _ in range(5):
        refine_center(rgb, box, max_side, config.RING_REFINE_MAX_CONTOURS, config.RING_REFINE_MAX_SAMPLES)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = refine_center(rgb, box, max_side, config.RING_REFINE_MAX_CONTOURS, config.RING_REFINE_MAX_SAMPLES)
        times.append((time.perf_counter() - start) * 1000)
    return result, {"median_ms": round(float(np.median(times)), 2),
                    "p95_ms": round(float(np.percentile(times, 95)), 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--box", nargs=4, type=float, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "center_trial")
    parser.add_argument("--max-side", type=int, default=config.RING_REFINE_MAX_SIDE)
    args = parser.parse_args()
    if args.image and not args.box:
        parser.error("--image requires --box in original image pixels")
    cv2.setNumThreads(1)
    report = {"timing_platform": "PC; estimator only, excludes YOLO/capture/UART",
              "max_side": args.max_side, "synthetic": []}
    for digit in "123":
        result, timing = measure(sample(digit), BOX, args.max_side)
        report["synthetic"].append({"digit": digit, "result": result, **timing,
            "bbox_error_px": round(float(np.hypot(BOX["x"] + BOX["w"]/2 - 355,
                                                  BOX["y"] + BOX["h"]/2 - 220)), 2),
            "refined_error_px": None if result is None else round(float(np.hypot(
                result["center"]["x"] - 355, result["center"]["y"] - 220)), 2)})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image:
        bgr = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cannot decode image")
        box = dict(zip(("x", "y", "w", "h"), args.box))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result, timing = measure(rgb, box, args.max_side)
        report["photo"] = {"path": str(args.image), "box": box, "result": result,
                           "accuracy": "unknown: no ground truth", **timing}
        cx, cy = round(box["x"] + box["w"] / 2), round(box["y"] + box["h"] / 2)
        cv2.drawMarker(bgr, (cx, cy), (0, 180, 255), cv2.MARKER_CROSS, 30, 2)
        if result:
            center = result["center"]
            cv2.drawMarker(bgr, (round(center["x"]), round(center["y"])), (0, 220, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.imencode(".jpg", bgr)[1].tofile(str(args.output / "photo_comparison.jpg"))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
