"""Offline top-face ellipse measurement for the 2026-10-07 grasp photos."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PHOTO_SETTINGS = {
    "3.jpg": ((90, 35, 20), (145, 255, 255), (1080, 150, 1650, 650)),
    "4.jpg": ((18, 35, 20), (42, 255, 255), (1080, 150, 1650, 650)),
    "5.jpg": ((0, 25, 20), (179, 255, 255), (1080, 150, 1650, 650)),
}


def measure(path):
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (1440, 2560):
        raise ValueError("Expected a 2560x1440 calibration photograph")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    name = Path(path).name
    lower, upper, (x1, y1, x2, y2) = PHOTO_SETTINGS[name]
    mask = cv2.inRange(hsv, lower, upper)
    roi = np.zeros(mask.shape, np.uint8)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = [c for c in contours if cv2.contourArea(c) > 50000]
    if not candidates:
        raise ValueError("top-face contour not found in %s" % name)
    points = max(candidates, key=cv2.contourArea).reshape(-1, 2)
    # The lower neck is not part of the top face. Keep the upper circular arc.
    rough = np.mean(points, axis=0)
    points = points[points[:, 1] < rough[1] + 65]
    ellipse = cv2.fitEllipse(points.astype(np.float32))
    center, axes, angle = ellipse
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    local = (points - center) @ rotation
    residual = (np.sqrt(np.sum((local / (np.array(axes) / 2)) ** 2, axis=1)) - 1)
    residual *= np.mean(axes) / 2
    return frame, points, ellipse, float(np.sqrt(np.mean(residual ** 2)))


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(__file__).resolve().parent / "calibration_20261007"
    output.mkdir(exist_ok=True)
    measurements = []
    previews = []
    for name in ("3.jpg", "4.jpg", "5.jpg"):
        path = root / "picture" / name
        frame, points, ellipse, residual = measure(path)
        row = {"photo": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "size": [2560, 1440], "center": list(ellipse[0]),
               "ellipse_axes": list(ellipse[1]), "arc_fit_rms_px": residual}
        measurements.append(row)
        cv2.ellipse(frame, ellipse, (0, 0, 255), 3)
        frame[points[:, 1], points[:, 0]] = (0, 255, 0)
        center = tuple(int(round(v)) for v in ellipse[0])
        cv2.drawMarker(frame, center, (255, 0, 255), cv2.MARKER_CROSS, 36, 3)
        crop = frame[80:850, 1000:1800].copy()
        cv2.putText(crop, "%s  (%.2f, %.2f)" % (name, *ellipse[0]),
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        previews.append(cv2.resize(crop, (600, 578)))
    centers = np.array([row["center"] for row in measurements])
    report = {"measurements": measurements, "mean_center": centers.mean(axis=0).tolist(),
              "sample_std_px": centers.std(axis=0, ddof=1).tolist(),
              "center_span_px": np.ptp(centers, axis=0).tolist(),
              "runtime_mapping": "Default centered crop, pending live-frame verification",
              "mapping_source": "https://wiki.sipeed.com/maixpy/doc/zh/vision/camera.html",
              "crop_xywh": [320, 0, 1920, 1440],
              "runtime_size": [640, 480],
              "runtime_center": ((centers.mean(axis=0) - [320, 0]) / 3).tolist()}
    (output / "measurement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cv2.imencode(".jpg", np.hstack(previews))[1].tofile(output / "measured_centers.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
