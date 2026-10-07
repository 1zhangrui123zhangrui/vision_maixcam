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
            # A target may be partially outside the frame. In target-ring mode
            # the center estimate is still useful, so do not reject it merely
            # because the YOLO box touches an edge.
            reject_edge = item["touches_image_edge"] and result["mode"] != "TARGET_RING"
            # A detected center is usable after temporal confirmation.  The
            # geometric pass improves the point when available; if the ring
            # is partially occluded, the YOLO center remains a valid coarse
            # center and is still allowed to reach the controller.
            center_bad = False
            if reject_edge or item["too_small"] or item["ambiguous"] or center_bad:
                continue
            key = item["class_id"]
            center = item["center"]
            prior = self.previous.get(key)
            count, anchor = 1, center
            if prior:
                anchor, old_count, last_center = prior
                anchor_drift = (center["x"] - anchor["x"]) ** 2 + (center["y"] - anchor["y"]) ** 2
                frame_drift = (center["x"] - last_center["x"]) ** 2 + (center["y"] - last_center["y"]) ** 2
                if (anchor_drift <= config.MAX_CENTER_DRIFT ** 2
                        and frame_drift < (config.MAX_CENTER_DRIFT * 0.5) ** 2):
                    count = min(config.CONFIRM_FRAMES, old_count + 1)
                else:
                    anchor = center
            # A newly reset track needs one stable frame before it can be
            # confirmed; a slow drift is accepted only when each frame stays
            # close to the previous anchor.
            current[key] = (anchor, count, center)
            item["confirmed"] = item["usable"] = count >= config.CONFIRM_FRAMES
        self.previous, self.last_ms = current, now_ms
        result["status"] = "OK" if any(i["usable"] for i in items) else ("UNCONFIRMED" if items else "NO_TARGET")
        return result
