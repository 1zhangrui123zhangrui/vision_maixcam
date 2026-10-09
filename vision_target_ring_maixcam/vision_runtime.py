"""Competition state, pixel reference and fresh recognition results."""

import gc
import time
import config
from calibration import PixelReference
from material_recognizer import MaterialMaix
from maix_target_ring import TargetRingMaix
from quality import Confirmation
from vision_protocol import MODEL_BY_MODE, ResultPacketEncoder, parse_command, newer


def monotonic_ms():
    return int(time.monotonic() * 1000)


class VisionRuntime:
    def __init__(self, mode=config.START_MODE, factories=None):
        if mode not in MODEL_BY_MODE:
            raise ValueError("unknown mode")
        self.mode = mode
        self.factories = factories or {"MATERIAL": MaterialMaix, "TARGET_RING": TargetRingMaix}
        self.reference = PixelReference()
        self.confirmation = Confirmation()
        self.encoder = ResultPacketEncoder()
        self.frame_id = 0
        self.last_command_seq = None
        self.recognizer = None
        self.loaded_model = None
        self.generation = 0

    def _load(self):
        model = MODEL_BY_MODE[self.mode]
        if model == "NONE":
            return
        if self.loaded_model != model:
            self.recognizer = None
            self.loaded_model = None
            gc.collect()
            # Keep only one NPU model resident. A load failure leaves no usable
            # result; the loop sends ERROR rather than using the previous model.
            self.recognizer = self.factories[model](confidence=config.CONFIDENCE, iou=config.IOU)
            self.loaded_model = model

    def handle_command(self, raw):
        command = parse_command(raw)
        if command["seq"] is not None and not newer(command["seq"], self.last_command_seq):
            return False
        if command["command"] == "SET_REFERENCE":
            self.reference.set(command["x"], command["y"])
        else:
            # Accept both the documented string mode and numeric JSON values
            # commonly emitted by MCU firmware.
            self.mode = str(command["mode"])
            # A mode command is a fresh acquisition request.  Do not retain a
            # previously loaded detector when the caller explicitly switches
            # back to that mode.
            self.recognizer = None
            self.loaded_model = None
            if self.mode == "IDLE":
                gc.collect()
        if command["seq"] is not None:
            self.last_command_seq = command["seq"]
        self.generation += 1
        self.confirmation.reset()
        return True

    def empty_result(self, status, timestamp_ms=None):
        self.confirmation.reset()
        model = MODEL_BY_MODE[self.mode]
        result = {"mode": model, "status": status, "materials": [], "targets": {},
                  "image_size": {"width": config.WIDTH, "height": config.HEIGHT}}
        if model == "TARGET_RING":
            result["targets"] = {"target_%d" % i: {"detected": False} for i in (1, 2, 3)}
        return self._stamp(result, monotonic_ms() if timestamp_ms is None else timestamp_ms)

    def _stamp(self, result, timestamp_ms):
        result = self.reference.add_to_result(result)
        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
        result.update(frame_id=self.frame_id, timestamp_ms=timestamp_ms,
                      stage=self.mode, generation=self.generation)
        return result

    def detect(self, frame, timestamp_ms=None):
        stamp = monotonic_ms() if timestamp_ms is None else timestamp_ms
        if MODEL_BY_MODE[self.mode] == "NONE":
            return self.empty_result("IDLE", stamp)
        if frame is None or frame.width() != config.WIDTH or frame.height() != config.HEIGHT:
            raise ValueError("camera frame does not match calibration resolution")
        self._load()
        result = self.recognizer.detect(frame)
        if result["mode"] != MODEL_BY_MODE[self.mode]:
            raise ValueError("recognizer returned wrong model type")
        if len(result.get("materials", [])) > config.MAX_OBJECTS:
            return self.empty_result("OVERFLOW", stamp)
        self.confirmation.apply(result, stamp)
        return self._stamp(result, stamp)

    def draw(self, frame, result):
        from maix import image

        if self.recognizer is not None and result["status"] not in ("ERROR", "IDLE", "OVERFLOW"):
            self.recognizer.draw(frame, result)
        # White cross: image center. Material reference is magenta; target-ring
        # reference is cyan so the two calibration points cannot be confused.
        reference = (self.reference.to_dict() if self.mode == "1" else
                     {"x": config.TARGET_REFERENCE_X, "y": config.TARGET_REFERENCE_Y})
        reference_rgb = (255, 0, 255) if self.mode == "1" else (0, 255, 255)
        for x, y, rgb in ((config.WIDTH // 2, config.HEIGHT // 2, (255, 255, 255)),
                          (int(reference["x"]), int(reference["y"]), reference_rgb)):
            color = image.Color.from_rgb(*rgb)
            frame.draw_line(max(0, x - 10), y, min(config.WIDTH - 1, x + 10), y, color, thickness=2)
            frame.draw_line(x, max(0, y - 10), x, min(config.HEIGHT - 1, y + 10), color, thickness=2)
        white = image.Color.from_rgb(255, 255, 255)
        frame.draw_string(2, 2, "MODE %s" % self.mode, white, scale=1)
        frame.draw_string(2, 20, "STATUS %s REF(%d,%d)" % (
            result["status"], reference["x"], reference["y"]), white, scale=1)
        return frame

    def packet(self, result, now_ms=None):
        if result["generation"] != self.generation or result["stage"] != self.mode:
            raise ValueError("cannot send a result from before a state/reference change")
        return self.encoder.encode(result, monotonic_ms() if now_ms is None else now_ms)
