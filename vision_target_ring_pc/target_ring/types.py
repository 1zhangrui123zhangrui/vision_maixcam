from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(float(self.x), 2), "y": round(float(self.y), 2)}


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
        }


@dataclass(frozen=True)
class TargetDetection:
    label: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    bbox_center: Point
    center: Point
    center_source: str
    touches_image_edge: bool
    ring_count: int = 0
    center_quality: float = 0.0
    center_usable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": True,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "bbox_center": self.bbox_center.to_dict(),
            "center": self.center.to_dict(),
            "center_source": self.center_source,
            "touches_image_edge": self.touches_image_edge,
            "ring_count": self.ring_count,
            "center_quality": round(float(self.center_quality), 4),
            "center_usable": self.center_usable,
        }


@dataclass(frozen=True)
class TargetRingResult:
    image_width: int
    image_height: int
    targets: dict[str, TargetDetection | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "TARGET_RING",
            "image_size": {"width": self.image_width, "height": self.image_height},
            "targets": {
                label: (detection.to_dict() if detection is not None else {"detected": False})
                for label, detection in self.targets.items()
            },
        }
