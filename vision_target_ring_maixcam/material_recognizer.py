"""Material top-face detection. No ring model or competition state logic."""

from detection import load_model, geometry, flag_ambiguity, draw_objects

LABELS = ("black", "blue", "green", "light_blue", "red", "yellow")
COLOR_IDS = {"red": 1, "yellow": 2, "blue": 3, "green": 4, "black": 5, "light_blue": 6}


class MaterialMaix:
    def __init__(self, model_path=None, confidence=0.45, iou=0.45):
        self.detector = load_model("model_9657.mud", LABELS, model_path)
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
                label = LABELS[item["class_id"]]
                item.update(color=label, color_id=COLOR_IDS[label], center_source="top_bbox_estimate")
                items.append(item)
        flag_ambiguity(items)
        items.sort(key=lambda item: (item["color_id"], -item["confidence"]))
        return {"mode": "MATERIAL", "image_size": {"width": width, "height": height},
                "materials": items}

    def draw(self, frame, result):
        return draw_objects(frame, result["materials"])
