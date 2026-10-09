from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np

from .recognizer import TargetRingRecognizer
from .types import TargetRingResult


class VisionMode(str, Enum):
    TARGET_RING = "TARGET_RING"
    MATERIAL = "MATERIAL"


class VisionController:
    """Command-facing switch shared by the controller and future serial layer."""

    def __init__(self, *, target_ring: TargetRingRecognizer) -> None:
        self.target_ring = target_ring
        self.mode = VisionMode.TARGET_RING

    def handle_command(self, command: str | Mapping[str, Any]) -> dict[str, Any]:
        payload = self._parse_command(command)
        name = str(payload.get("command", "")).upper()
        if name == "SET_MODE":
            try:
                self.mode = VisionMode(str(payload["mode"]).upper())
            except (KeyError, ValueError) as exc:
                raise ValueError("mode must be TARGET_RING or MATERIAL") from exc
            return {"ok": True, "command": "SET_MODE", "mode": self.mode.value}
        if name == "GET_MODE":
            return {"ok": True, "command": "GET_MODE", "mode": self.mode.value}
        if name == "DETECT":
            return {"ok": True, "command": "DETECT", "mode": self.mode.value}
        raise ValueError(f"unsupported vision command: {name or '<empty>'}")

    def process_frame(self, frame: np.ndarray) -> TargetRingResult:
        if self.mode is VisionMode.TARGET_RING:
            result = self.target_ring.detect(frame)
            assert isinstance(result, TargetRingResult)
            return result
        raise NotImplementedError("material recognizer is not implemented yet")

    @staticmethod
    def _parse_command(command: str | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(command, Mapping):
            return command
        text = command.strip()
        if not text:
            raise ValueError("empty vision command")
        # Simple text commands make bench testing possible without a serial
        # dependency. Examples: SET_MODE TARGET_RING, GET_MODE.
        parts = text.split()
        if parts[0].upper() == "SET_MODE" and len(parts) == 2:
            return {"command": parts[0], "mode": parts[1]}
        if len(parts) == 1:
            return {"command": parts[0]}
        raise ValueError(f"cannot parse vision command: {command!r}")
