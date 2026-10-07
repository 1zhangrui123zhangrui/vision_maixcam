"""Minimal newline JSON result packets with sequence numbers and no ACK."""

import json
import config

MODEL_BY_MODE = {
    "IDLE": "NONE",
    "1": "MATERIAL",
    "2": "TARGET_RING",
}
MODES = tuple(MODEL_BY_MODE)
COMMAND_MODE_NAMES = {"1": "物料", "2": "靶环"}


def parse_command(line):
    if isinstance(line, bytes):
        line = line.decode("ascii")
    if isinstance(line, str) and line.strip() in ("1", "2"):
        return {"command": "SET_MODE", "mode": line.strip(), "seq": None}
    command = json.loads(line) if isinstance(line, str) else line
    if not isinstance(command, dict):
        raise ValueError("command must be a JSON object")
    seq = command.get("seq")
    if seq is not None and (type(seq) is not int or not 0 <= seq <= 0xFFFFFFFF):
        raise ValueError("command requires uint32 seq")
    name = command.get("command")
    if name == "SET_MODE":
        if str(command.get("mode")) not in MODEL_BY_MODE:
            raise ValueError("unknown mode")
    elif name == "SET_REFERENCE":
        if "x" not in command or "y" not in command:
            raise ValueError("reference requires x and y")
    else:
        raise ValueError("unknown command")
    return command


def newer(sequence, previous):
    return previous is None or 0 < ((sequence - previous) & 0xFFFFFFFF) < 0x80000000


class ResultPacketEncoder:
    """Encode only the packet sequence and object ids/pixel offsets."""

    def __init__(self, sequence=0):
        self.sequence = int(sequence) & 0xFFFFFFFF

    def encode(self, result, now_ms):
        stale = now_ms - result["timestamp_ms"] > config.MAX_RESULT_AGE_MS
        items = (result.get("materials", []) if result["mode"] == "MATERIAL" else
                 [v for v in result.get("targets", {}).values() if v["detected"]])
        compact_items = []
        for item in ([] if stale else items):
            # The compact protocol has no flags/status field. Only emit a
            # center after the existing multi-frame confirmation accepts it.
            if not item.get("usable", False):
                continue
            object_id = item.get("color_id", item.get("target_id"))
            compact_items.append([object_id, item["offset"]["x"], item["offset"]["y"]])
        if len(compact_items) > config.MAX_OBJECTS:
            raise ValueError("result exceeds object limit")
        packet = {
            "seq": self.sequence,
            "items": compact_items,
        }
        data = (json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if len(data) > 1000:
            raise ValueError("packet exceeds UART bandwidth budget")
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return data
