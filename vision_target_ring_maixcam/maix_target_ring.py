"""Numbered ring recognition with bounded-ROI center refinement."""

import time
import config
from detection import load_model, geometry, flag_ambiguity, draw_objects

LABELS = ("target_1", "target_2", "target_3")


class TargetRingMaix:
    def __init__(self, model_path=None, confidence=0.45, iou=0.45):
        self.detector = load_model("model_9728.mud", LABELS, model_path)
        self.confidence, self.iou = confidence, iou
        self.refine = None
        self.warned = False
        self.frame_index = 0
        self.tracks = {}
        self.attempts = {}
        if config.RING_REFINE_ENABLED:
            try:
                from ring_center import refine_center
                self.refine = refine_center
            except ImportError as exc:
                print("Ring refinement unavailable, using boxes:", exc)

    def detect(self, frame):
        from maix import image

        started = time.monotonic()
        self.frame_index += 1
        width, height = frame.width(), frame.height()
        objects = self.detector.detect(frame, conf_th=self.confidence,
                                       iou_th=self.iou, fit=image.Fit.FIT_CONTAIN)
        items = []
        for obj in objects:
            item = geometry(obj, width, height, self.confidence, len(LABELS))
            if item is not None:
                item.update(detected=True, label=LABELS[item["class_id"]],
                            target_id=item["class_id"] + 1,
                            bbox_center=dict(item["center"]), center_source="bbox_maix",
                            center_quality=0.0, ring_count=0, center_usable=False)
                items.append(item)
        flag_ambiguity(items)
        targets = {label: {"detected": False} for label in LABELS}
        for item in sorted(items, key=lambda item: item["confidence"]):
            targets[item["label"]] = item
        refine_started = time.monotonic()
        eligible = [item for item in targets.values()
                    if item["detected"] and not item["ambiguous"] and not item["too_small"]]
        visible = {item["class_id"] for item in eligible}
        self.tracks = {key: track for key, track in self.tracks.items() if key in visible}
        self.attempts = {key: value for key, value in self.attempts.items() if key in visible}
        # Expensive ellipse fitting is periodic. Between fits, move the last
        # precise center by the detector-box displacement to keep control data
        # responsive while the robot is making small corrections.
        for item in eligible:
            track = self.tracks.get(item["class_id"])
            if track is None:
                continue
            dx = item["bbox_center"]["x"] - track["bbox_center"]["x"]
            dy = item["bbox_center"]["y"] - track["bbox_center"]["y"]
            cx, cy = track["center"]["x"] + dx, track["center"]["y"] + dy
            invalid = (self.frame_index - track["frame"] > config.RING_TRACK_MAX_AGE
                       or (started - track["time"]) * 1000 > config.RING_TRACK_MAX_AGE_MS
                       or dx * dx + dy * dy > config.RING_TRACK_MAX_SHIFT ** 2
                       or any(abs(item["bbox"][axis] / track["bbox"][axis] - 1)
                              > config.RING_TRACK_MAX_SIZE_CHANGE for axis in ("w", "h"))
                       or item["touches_image_edge"] != track["edge"]
                       or not (0 <= cx < width and 0 <= cy < height))
            if invalid:
                del self.tracks[item["class_id"]]
                continue
            item["center"] = {"x": round(cx, 2), "y": round(cy, 2)}
            item.update(center_source="tracked", center_quality=track["quality"],
                        ring_count=track["ring_count"], center_usable=True)
        due = [item for item in eligible if self.frame_index - self.attempts.get(
            item["class_id"], -config.RING_REFINE_INTERVAL) >= config.RING_REFINE_INTERVAL]
        # One ROI per frame avoids three simultaneous expensive fits. Oldest
        # attempt first provides fair retries even when one target never fits.
        if self.refine is not None and due:
            item = min(due, key=lambda obj: self.attempts.get(obj["class_id"], -1000000))
            self.attempts[item["class_id"]] = self.frame_index
            try:
                rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
                refined = self.refine(rgb, item["bbox"], config.RING_REFINE_MAX_SIDE,
                                      config.RING_REFINE_MAX_CONTOURS,
                                      config.RING_REFINE_MAX_SAMPLES)
                if refined is not None:
                    item.update(refined)
                    self.tracks[item["class_id"]] = {
                        "center": dict(item["center"]), "bbox": dict(item["bbox"]),
                        "bbox_center": dict(item["bbox_center"]),
                        "quality": item["center_quality"], "ring_count": item["ring_count"],
                        "frame": self.frame_index, "time": started,
                        "edge": item["touches_image_edge"],
                    }
            except Exception as exc:
                if not self.warned:
                    print("Ring refinement failed, using boxes:", exc)
                    self.warned = True
        finished = time.monotonic()
        # A box fallback is not cached as a precise track. It remains eligible
        # for later geometry correction instead of accumulating box bias.
        return {"mode": "TARGET_RING", "image_size": {"width": width, "height": height},
                "targets": targets, "refine_ms": round((finished - refine_started) * 1000, 1),
                "detect_ms": round((finished - started) * 1000, 1)}

    def draw(self, frame, result):
        from maix import image

        draw_objects(frame, [item for item in result["targets"].values() if item["detected"]])
        states = ["%d:%s" % (item["target_id"],
                            "ELL" if item["center_source"] == "multi_ellipse" else
                            "TRK" if item["center_source"] == "tracked" else "BOX")
                  for item in result["targets"].values() if item["detected"]]
        frame.draw_string(2, 38, "DET %.0fms GEO %.0fms %s" % (
            result["detect_ms"], result["refine_ms"], " ".join(states)),
            image.Color.from_rgb(255, 255, 255), scale=1)
        return frame
