from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import BoundingBox, Point


@dataclass(frozen=True)
class CenterEstimate:
    point: Point
    source: str
    ring_count: int
    quality: float
    usable: bool


def estimate_center(image: np.ndarray, bbox: BoundingBox) -> CenterEstimate:
    """Estimate a shared center from concentric ellipse edges inside a YOLO ROI."""
    fallback = CenterEstimate(bbox.center, "bbox", 0, 0.0, False)
    height, width = image.shape[:2]
    pad_x, pad_y = bbox.width * 0.10, bbox.height * 0.10
    x1, y1 = max(0, int(bbox.x1 - pad_x)), max(0, int(bbox.y1 - pad_y))
    x2, y2 = min(width, int(np.ceil(bbox.x2 + pad_x))), min(height, int(np.ceil(bbox.y2 + pad_y)))
    if x2 - x1 < 48 or y2 - y1 < 40:
        return fallback

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    edges = cv2.Canny(gray, 35, 110, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    roi_center = np.array([bbox.center.x - x1, bbox.center.y - y1], dtype=np.float64)
    min_axis = max(8.0, min(bbox.width, bbox.height) * 0.10)
    max_axis = max(bbox.width, bbox.height) * 0.72

    candidates: list[tuple[np.ndarray, float, float, float]] = []
    for contour in contours:
        if len(contour) < 24:
            continue
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (diam_a, diam_b), _ = ellipse
        major, minor = max(diam_a, diam_b) / 2.0, min(diam_a, diam_b) / 2.0
        if minor < min_axis or major > max_axis or minor / major < 0.28:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        # Algebraic ellipse residual, normalized by the fitted ellipse scale.
        angle = np.deg2rad(ellipse[2])
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        radial = np.sqrt((u / (diam_a / 2.0)) ** 2 + (v / (diam_b / 2.0)) ** 2)
        residual = float(np.median(np.abs(radial - 1.0)))
        coverage = _angular_coverage(u / major, v / minor)
        center = np.array([cx, cy], dtype=np.float64)
        if np.linalg.norm(center - roi_center) > min(bbox.width, bbox.height) * 0.24:
            continue
        if residual > 0.055 or coverage < 0.32:
            continue
        quality = coverage / (1.0 + 20.0 * residual)
        candidates.append((center, major, quality, residual))

    if len(candidates) < 2:
        return fallback

    # Cluster ellipse centers; digit strokes and unrelated contours rarely
    # produce several differently-sized ellipses with one common center.
    points = np.array([item[0] for item in candidates])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    tolerance = max(3.0, min(bbox.width, bbox.height) * 0.045)
    best_indices = np.where((distances <= tolerance).sum(axis=1) >= 2)[0]
    if len(best_indices) == 0:
        return fallback
    seed = points[best_indices].mean(axis=0)
    inliers = [item for item in candidates if np.linalg.norm(item[0] - seed) <= tolerance]
    # Require distinct ring scales; duplicate inner/outer edge contours of one
    # printed band must not count as independent evidence.
    inliers.sort(key=lambda item: item[1], reverse=True)
    distinct: list[tuple[np.ndarray, float, float, float]] = []
    for item in inliers:
        if all(abs(item[1] - kept[1]) >= max(2.5, kept[1] * 0.045) for kept in distinct):
            distinct.append(item)
    if len(distinct) < 2:
        return fallback

    centers = np.array([item[0] for item in distinct])
    center_median = np.median(centers, axis=0)
    errors = np.linalg.norm(centers - center_median, axis=1)
    robust = errors <= tolerance
    distinct = [item for item, keep in zip(distinct, robust) if keep]
    if len(distinct) < 2:
        return fallback
    weights = np.array([item[2] * np.sqrt(item[1]) for item in distinct])
    fused = np.average(np.array([item[0] for item in distinct]), axis=0, weights=weights)
    spread = float(np.median(np.linalg.norm(np.array([item[0] for item in distinct]) - fused, axis=1)))
    quality = float(np.mean([item[2] for item in distinct]) * np.exp(-spread / max(tolerance, 1.0)))
    if spread > tolerance or np.linalg.norm(fused - roi_center) > min(bbox.width, bbox.height) * 0.20:
        return fallback
    return CenterEstimate(Point(float(fused[0] + x1), float(fused[1] + y1)),
                          "multi_ellipse", len(distinct), quality, quality >= 0.30)


def _angular_coverage(x: np.ndarray, y: np.ndarray, bins: int = 24) -> float:
    angles = np.arctan2(y, x)
    occupied = np.unique(np.floor((angles + np.pi) * bins / (2.0 * np.pi)).astype(int) % bins)
    return len(occupied) / bins
