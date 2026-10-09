from __future__ import annotations

from typing import Any, Protocol


class VisionResultTransport(Protocol):
    """Future serial/network adapters implement this small interface."""

    def send(self, payload: dict[str, Any]) -> None:
        ...


class NullVisionResultTransport:
    """No-op transport for local tests; deliberately does not open a port."""

    def send(self, payload: dict[str, Any]) -> None:
        return None
