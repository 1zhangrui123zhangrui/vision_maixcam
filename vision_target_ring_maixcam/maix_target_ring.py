"""Numbered ring detection. No material model or competition state logic."""

from detection import load_model, geometry, flag_ambiguity, draw_objects

LABELS = ("target_1", "target_2", "target_3")


class TargetRingMaix:
    def __init__(self, model_path=None, confidence=0.45, iou=0.45):
        self.detector = load_model("model_9621.mud", LABELS, model_path)
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
                item.update(detected=True, label=LABELS[item["class_id"]], target_id=item["class_id"] + 1)
                items.append(item)
        flag_ambiguity(items)
        targets = {label: {"detected": False} for label in LABELS}
        for item in sorted(items, key=lambda item: item["confidence"]):
            targets[item["label"]] = item
        return {"mode": "TARGET_RING", "image_size": {"width": width, "height": height},
                "targets": targets}

    def draw(self, frame, result):
        return draw_objects(frame, [item for item in result["targets"].values() if item["detected"]])
