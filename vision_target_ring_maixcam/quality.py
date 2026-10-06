"""Consecutive-frame confirmation without smoothing or reusing missing centers."""

import config


class Confirmation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = {}
        self.last_ms = None

    def apply(self, result, now_ms):
        if self.last_ms is not None and now_ms - self.last_ms > config.MAX_FRAME_GAP_MS:
            self.reset()
        items = (result.get("materials", []) if result["mode"] == "MATERIAL" else
                 [v for v in result.get("targets", {}).values() if v["detected"]])
        current = {}
        for item in items:
            item["confirmed"] = item["usable"] = False
            if item["touches_image_edge"] or item["too_small"] or item["ambiguous"]:
                continue
            key = item["class_id"]
            center = item["center"]
            prior = self.previous.get(key)
            count, anchor = 1, center
            if prior:
                anchor, old_count = prior
                drift = (center["x"] - anchor["x"]) ** 2 + (center["y"] - anchor["y"]) ** 2
                if drift <= config.MAX_CENTER_DRIFT ** 2:
                    count = min(config.CONFIRM_FRAMES, old_count + 1)
                else:
                    anchor = center
            current[key] = (anchor, count)
            item["confirmed"] = item["usable"] = count >= config.CONFIRM_FRAMES
        self.previous, self.last_ms = current, now_ms
        result["status"] = "OK" if any(i["usable"] for i in items) else ("UNCONFIRMED" if items else "NO_TARGET")
        return result
