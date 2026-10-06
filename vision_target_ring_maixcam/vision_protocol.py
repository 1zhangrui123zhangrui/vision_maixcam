"""Version 1: bounded newline JSON, sequence numbers, no ACK packets."""

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
    """Encode explicit material/target payloads plus a compact compatibility list."""

    def __init__(self, sequence=0):
        self.sequence = int(sequence) & 0xFFFFFFFF

    def encode(self, result, now_ms):
        stale = now_ms - result["timestamp_ms"] > config.MAX_RESULT_AGE_MS
        items = (result.get("materials", []) if result["mode"] == "MATERIAL" else
                 [v for v in result.get("targets", {}).values() if v["detected"]])
        objects = []
        named = []
        for item in ([] if stale else items):
            flags = (int(item["usable"]) | int(item["touches_image_edge"]) << 1
                     | int(item["ambiguous"]) << 2 | int(item["too_small"]) << 3
                     | int(item["confirmed"]) << 4)
            object_id = item.get("color_id", item.get("target_id"))
            entry = {"id": object_id, "confidence": item["confidence"],
                     "x": item["center"]["x"], "y": item["center"]["y"],
                     "dx": item["offset"]["x"], "dy": item["offset"]["y"],
                     "flags": flags}
            named.append(entry)
            objects.append([object_id, item["confidence"], item["center"]["x"],
                            item["center"]["y"], item["offset"]["x"],
                            item["offset"]["y"], flags])
        if len(objects) > config.MAX_OBJECTS:
            raise ValueError("result exceeds object limit")
        packet = {
            "v": 1, "seq": self.sequence, "type": "VISION_RESULT",
            "mode": result["stage"], "model": result["mode"],
            "status": "STALE" if stale else result["status"],
            "frame_id": result["frame_id"], "t_ms": result["timestamp_ms"],
            "age_ms": max(0, now_ms - result["timestamp_ms"]),
            "size": [result["image_size"]["width"], result["image_size"]["height"]],
            "ref": [result["reference"]["x"], result["reference"]["y"]],
            "objects": objects,
        }
        if result["mode"] == "MATERIAL":
            packet["materials"] = named
        elif result["mode"] == "TARGET_RING":
            by_id = {entry["id"]: entry for entry in named}
            packet["targets"] = [by_id.get(i, {"id": i, "detected": False,
                              "confidence": 0, "x": 0, "y": 0,
                              "dx": 0, "dy": 0, "flags": 0}) for i in (1, 2, 3)]
        data = (json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if len(data) > 1000:
            raise ValueError("packet exceeds UART bandwidth budget")
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return data
