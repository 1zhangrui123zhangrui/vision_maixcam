"""Bounded ROI ellipse refinement for target-ring centers.

Ellipse centers approximate the projected physical center only under weak
perspective. This is not lens correction or projective plane calibration.
"""

import math

import cv2
import numpy as np


def refine_center(rgb, box, max_side=240, max_contours=24, max_samples=96):
    """Return verified local geometry, or None to retain the YOLO estimate."""
    height, width = rgb.shape[:2]
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    if min(w, h) < 32:
        return None
    x0, y0 = max(0, int(x - w * .10)), max(0, int(y - h * .10))
    x1, y1 = min(width, math.ceil(x + w * 1.10)), min(height, math.ceil(y + h * 1.10))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = rgb[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(roi.shape[:2]))
    if scale < 1:
        roi = cv2.resize(roi, (round(roi.shape[1] * scale), round(roi.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    sx, sy = roi.shape[1] / (x1 - x0), roi.shape[0] / (y1 - y0)
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), (3, 3), .7)
    edges = cv2.Canny(gray, 35, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    origin = np.array([(x + w / 2 - x0 + .5) * sx - .5,
                       (y + h / 2 - y0 + .5) * sy - .5])
    span = min(w * sx, h * sy)
    tolerance = max(1.5, span * .035)
    candidates = []
    # Limit both contour count and samples; fitting stays in compiled OpenCV.
    for contour in sorted(contours, key=len, reverse=True)[:max_contours]:
        if len(contour) < 24:
            continue
        sampled = np.ascontiguousarray(contour[::max(1, math.ceil(len(contour) / max_samples))], dtype=np.float32)
        try:
            (cx, cy), (da, db), angle = cv2.fitEllipse(sampled)
        except cv2.error:
            continue
        if not all(math.isfinite(v) for v in (cx, cy, da, db, angle)):
            continue
        major, minor = max(da, db) / 2, min(da, db) / 2
        if (minor < max(5, span * .16) or major > max(w * sx, h * sy) * .70
                or minor / major < .22):
            continue
        center = np.array([cx, cy])
        if np.linalg.norm(center - origin) > span * .28:
            continue
        points = sampled.reshape(-1, 2).astype(np.float64) - center
        radians = math.radians(angle)
        c, s = math.cos(radians), math.sin(radians)
        u = (points[:, 0] * c + points[:, 1] * s) / (da / 2)
        v = (-points[:, 0] * s + points[:, 1] * c) / (db / 2)
        residual = float(np.percentile(np.abs(np.hypot(u, v) - 1), 80))
        bins = (np.floor((np.arctan2(v, u) + np.pi) * 24 / (2 * np.pi)).astype(int) % 24)
        coverage = len(np.unique(bins)) / 24
        if residual > .045 or coverage < .65:
            continue
        candidates.append((center, major, minor, coverage / (1 + 25 * residual)))
    if len(candidates) < 2:
        return None
    centers = np.array([item[0] for item in candidates])
    neighbors = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2) <= tolerance
    best, best_weight = [], 0.0
    for neighborhood in neighbors:
        group = [item for item, keep in zip(candidates, neighborhood) if keep]
        group.sort(key=lambda item: item[3], reverse=True)
        distinct = []
        for item in group:
            if all(abs(item[1] - other[1]) > max(3, .10 * max(item[1], other[1])) for other in distinct):
                distinct.append(item)
        if len(distinct) < 2:
            continue
        ratios = [item[2] / item[1] for item in distinct]
        if max(ratios) - min(ratios) > .12:
            continue
        weight = sum(item[3] for item in distinct)
        if weight > best_weight:
            best, best_weight = distinct, weight
    if not best:
        return None
    fused = np.average(np.array([item[0] for item in best]), axis=0, weights=[item[3] for item in best])
    spread = max(float(np.linalg.norm(item[0] - fused)) for item in best)
    quality = best_weight / len(best) * math.exp(-spread / tolerance)
    # Invert OpenCV's pixel-center resize mapping, including half-pixel offsets.
    cx, cy = float((fused[0] + .5) / sx - .5 + x0), float((fused[1] + .5) / sy - .5 + y0)
    if quality < .35 or not (0 <= cx < width and 0 <= cy < height):
        return None
    return {"center": {"x": round(cx, 2), "y": round(cy, 2)},
            "center_source": "multi_ellipse", "center_quality": round(quality, 3),
            "ring_count": len(best), "center_usable": True}
