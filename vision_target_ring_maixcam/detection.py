"""Shared geometry and model loading, with no task/mode decisions."""

import math
import os
import config


def load_model(filename, labels, model_path=None):
    from maix import nn

    path = str(model_path) if model_path else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    detector = nn.YOLO11(model=path, dual_buff=False)
    if tuple(detector.labels) != tuple(labels):
        raise ValueError("model labels do not match recognizer: " + path)
    print("Model input:", filename, detector.input_width(), detector.input_height())
    return detector


def geometry(obj, width, height, confidence, class_count):
    def field(name, default):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    class_id = field("class_id", -1)
    if not isinstance(class_id, int) or not 0 <= class_id < class_count:
        return None
    x, y, w, h, score = (float(field(k, 0)) for k in ("x", "y", "w", "h", "score"))
    if not all(math.isfinite(v) for v in (x, y, w, h, score)):
        return None
    if w <= 0 or h <= 0 or not confidence <= score <= 1:
        return None
    raw_right, raw_bottom = x + w, y + h
    right, bottom = min(width, raw_right), min(height, raw_bottom)
    left, top = max(0, x), max(0, y)
    if right <= left or bottom <= top:
        return None
    visible_area = (right - left) * (bottom - top)
    visible_ratio = visible_area / max(w * h, 1e-6)
    edge = (x <= config.EDGE_MARGIN or y <= config.EDGE_MARGIN
            or x + w >= width - config.EDGE_MARGIN or y + h >= height - config.EDGE_MARGIN)
    small = min(right - left, bottom - top) < config.MIN_BOX_SIZE
    return {
        "class_id": class_id, "confidence": round(score, 4),
        "bbox": {"x": round(left, 2), "y": round(top, 2),
                 "w": round(right - left, 2), "h": round(bottom - top, 2)},
        "center": {"x": round((left + right) / 2, 2), "y": round((top + bottom) / 2, 2)},
        "center_source": "bbox_estimate", "touches_image_edge": edge,
        "visible_ratio": round(visible_ratio, 4),
        "too_small": small, "ambiguous": False,
    }


def flag_ambiguity(items):
    # Preserve competing candidates, but never silently select one for motion.
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            box_a, box_b = a["bbox"], b["bbox"]
            overlap_w = max(0, min(box_a["x"] + box_a["w"], box_b["x"] + box_b["w"])
                            - max(box_a["x"], box_b["x"]))
            overlap_h = max(0, min(box_a["y"] + box_a["h"], box_b["y"] + box_b["h"])
                            - max(box_a["y"], box_b["y"]))
            intersection = overlap_w * overlap_h
            union = box_a["w"] * box_a["h"] + box_b["w"] * box_b["h"] - intersection
            if a["class_id"] == b["class_id"] or intersection / max(union, 1e-6) > 0.4:
                a["ambiguous"] = b["ambiguous"] = True


def draw_objects(frame, items):
    from maix import image

    for item in items:
        box, center = item["bbox"], item["center"]
        rgb = (0, 220, 0) if item.get("usable") else (255, 180, 0)
        color = image.Color.from_rgb(*rgb)
        frame.draw_rect(int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"]), color, thickness=2)
        frame.draw_circle(int(center["x"]), int(center["y"]), 4, color, thickness=2)
        label = item.get("color", item.get("label", ""))
        offset = item.get("offset", {"x": 0, "y": 0})
        state = "OK" if item.get("usable") else "WAIT"
        reason = "" if item.get("usable") else "[" + item.get("wait_reason", "") + "]"
        text = "%s %.2f %s%s (%d,%d) d(%d,%d)" % (
            label, item["confidence"], state,
            reason,
            int(center["x"]), int(center["y"]),
            int(offset["x"]), int(offset["y"]),
        )
        frame.draw_string(max(0, int(box["x"])), max(0, int(box["y"]) - 18),
                          text, color, scale=1)
    return frame
