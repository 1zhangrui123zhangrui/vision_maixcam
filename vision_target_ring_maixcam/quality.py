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
            item["wait_reason"] = ""
            is_target = result["mode"] == "TARGET_RING"
            # A target may be partially outside the frame. In target-ring mode
            # the center estimate is still useful, so do not reject it merely
            # because the YOLO box touches an edge.
            # A box may touch the border while the material is still almost
            # fully visible. Reject only genuinely clipped material boxes.
            reject_edge = (item["touches_image_edge"]
                           and result["mode"] != "TARGET_RING"
                           and (config.MATERIAL_REJECT_EDGE or
                                item.get("visible_ratio", 0.0) < config.MATERIAL_MIN_VISIBLE_RATIO))
            # A detected center is usable after temporal confirmation. The
            # geometric pass improves the point when available; if the ring
            # is partial, the detector center remains a valid fallback.
            if reject_edge or item["too_small"] or item["ambiguous"]:
                item["wait_reason"] = ("EDGE" if reject_edge else
                                        "SMALL" if item["too_small"] else "AMB")
                continue
            # Geometry and tracking sources may alternate while the same
            # target is moving. They are one physical track, not two tracks.
            key = (item["class_id"], None if is_target else item.get("center_source"))
            center = item["center"]
            prior = self.previous.get(key)
            count, anchor = 1, center
            if prior:
                anchor, old_count, last_center = prior
                anchor_drift = (center["x"] - anchor["x"]) ** 2 + (center["y"] - anchor["y"]) ** 2
                frame_drift = (center["x"] - last_center["x"]) ** 2 + (center["y"] - last_center["y"]) ** 2
                max_step = (config.TARGET_MAX_FRAME_STEP if is_target else
                            config.MATERIAL_MAX_FRAME_STEP)
                step_ok = (frame_drift <= max_step ** 2 if is_target
                           else frame_drift < max_step ** 2)
                if step_ok and (is_target or anchor_drift <= config.MAX_CENTER_DRIFT ** 2):
                    needed = config.TARGET_CONFIRM_FRAMES if is_target else config.CONFIRM_FRAMES
                    count = min(needed, old_count + 1)
                else:
                    anchor = center
                    item["wait_reason"] = "MOVE"
            # Rings follow continuous motion; materials retain a fixed anchor.
            current[key] = (anchor, count, center)
            needed = config.TARGET_CONFIRM_FRAMES if is_target else config.CONFIRM_FRAMES
            item["confirmed"] = item["usable"] = count >= needed
            if not item["usable"] and not item["wait_reason"]:
                item["wait_reason"] = "STABLE %d/%d" % (count, needed)
        self.previous, self.last_ms = current, now_ms
        result["status"] = "OK" if any(i["usable"] for i in items) else ("UNCONFIRMED" if items else "NO_TARGET")
        return result
