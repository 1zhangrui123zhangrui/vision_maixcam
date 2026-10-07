"""Pixel reference and offset calculation for the fixed camera."""

from __future__ import annotations

import math
import config

# Gripped top-face center measured from the 2026-10-07 photos.
# Mapping assumes a centered 1920x1440 crop scaled to 640x480; no padding.
# This remains a pixel-space offset reference, not a millimetre/world map.
DEFAULT_REFERENCE_X = config.REFERENCE_X
DEFAULT_REFERENCE_Y = config.REFERENCE_Y


class PixelReference:
    def __init__(self, x=DEFAULT_REFERENCE_X, y=DEFAULT_REFERENCE_Y,
                 width=config.WIDTH, height=config.HEIGHT):
        self.width, self.height = width, height
        self.set(x, y)

    def set(self, x, y):
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("reference must contain numbers")
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)
                and 0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("reference must be inside the camera image")
        self.x = x
        self.y = y

    def to_dict(self):
        return {"x": round(self.x, 2), "y": round(self.y, 2)}

    def add_to_result(self, result):
        """Return a result with reference and per-object pixel offsets."""
        output = dict(result)
        output["reference"] = self.to_dict()
        output["coordinate_unit"] = "pixel"
        output["offset_convention"] = "center_minus_reference; +x right, +y down"
        if result.get("mode") == "MATERIAL":
            output["materials"] = [self._with_offset(item) for item in result.get("materials", [])]
        elif result.get("mode") == "TARGET_RING":
            output["targets"] = {
                label: self._with_offset(item) if item.get("detected") else item
                for label, item in result.get("targets", {}).items()
            }
        return output

    def _with_offset(self, item):
        value = dict(item)
        center = value["center"]
        value["offset"] = {
            "x": round(float(center["x"]) - self.x, 2),
            "y": round(float(center["y"]) - self.y, 2),
        }
        return value
