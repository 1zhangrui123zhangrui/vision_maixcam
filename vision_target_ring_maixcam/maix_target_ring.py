"""Numbered target-ring detection with a lightweight geometric center pass."""

import math
import config
from detection import load_model, geometry, flag_ambiguity, draw_objects

LABELS = ("target_1", "target_2", "target_3")


class TargetRingMaix:
    def __init__(self, model_path=None, confidence=0.45, iou=0.45):
        self.detector = load_model("model_9728.mud", LABELS, model_path)
        self.confidence, self.iou = confidence, iou

    def detect(self, frame):
        from maix import image

        width, height = frame.width(), frame.height()
        objects = self.detector.detect(frame, conf_th=self.confidence,
                                       iou_th=self.iou, fit=image.Fit.FIT_CONTAIN)
        items = []
        for obj in objects:
            item = geometry(obj, width, height, self.confidence, len(LABELS))
            if item is not None:
                center = _estimate_center(frame, item["bbox"], item["center"])
                item["center"] = {"x": round(center[0], 2), "y": round(center[1], 2)}
                item.update(detected=True, label=LABELS[item["class_id"]],
                            target_id=item["class_id"] + 1,
                            center_source="dark_roi" if center[2] else "bbox_maix",
                            center_quality=round(center[3], 3),
                            ring_count=1 if center[2] else 0,
                            center_usable=bool(center[2]))
                items.append(item)
        flag_ambiguity(items)
        targets = {label: {"detected": False} for label in LABELS}
        for item in sorted(items, key=lambda item: item["confidence"]):
            targets[item["label"]] = item
        return {"mode": "TARGET_RING", "image_size": {"width": width, "height": height},
                "targets": targets}

    def draw(self, frame, result):
        return draw_objects(frame, [item for item in result["targets"].values() if item["detected"]])


def _estimate_center(frame, box, fallback):
    """Estimate a center from a balanced dark-pixel mask in the YOLO ROI.

    The mask is sampled symmetrically around the detector center, so the
    printed number cannot by itself move the result arbitrarily.  If the mask
    is incomplete or asymmetric, the caller keeps the detector center as a
    deliberate fallback.  The PC implementation remains the precision
    multi-ellipse validator.
    """
    try:
        gray = frame.to_format(frame.Format.FMT_GRAYSCALE) if hasattr(frame, "Format") else frame
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
        pad_x, pad_y = w * config.RING_ROI_PADDING, h * config.RING_ROI_PADDING
        x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
        x1, y1 = min(frame.width(), int(x + w + pad_x)), min(frame.height(), int(y + h + pad_y))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return float(fallback["x"]), float(fallback["y"]), False, 0.0
        # Maix image supports get_pixel; sample on a coarse grid to keep the
        # per-frame cost bounded and independent of OpenCV availability.
        step = max(1, min(x1 - x0, y1 - y0) // 36)
        points = []
        for py in range(y0, y1, step):
            for px in range(x0, x1, step):
                value = gray.get_pixel(px, py)
                if isinstance(value, tuple):
                    value = value[0]
                if float(value) < config.RING_DARK_THRESHOLD:
                    points.append((px, py))
        if len(points) < config.RING_MIN_PIXELS:
            return float(fallback["x"]), float(fallback["y"]), False, 0.0
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        # Keep the geometric estimate close to YOLO ROI center.  A large shift
        # is usually caused by the printed digit or a neighboring object.
        distance = math.hypot(cx - fallback["x"], cy - fallback["y"])
        limit = min(w, h) * config.RING_CENTER_MAX_SHIFT_RATIO
        if distance > limit:
            return float(fallback["x"]), float(fallback["y"]), False, 0.0
        symmetry = max(0.0, 1.0 - distance / max(limit, 1.0))
        return cx, cy, True, symmetry
    except Exception:
        return float(fallback["x"]), float(fallback["y"]), False, 0.0
