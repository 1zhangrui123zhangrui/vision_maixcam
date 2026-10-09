"""Hardware-independent tests against the actual deployment modules."""

import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
DEVICE = Path(__file__).resolve().parents[1] / "vision_target_ring_maixcam"
sys.path.insert(0, str(DEVICE))

import config
from calibration import PixelReference
from detection import geometry, flag_ambiguity
from material_recognizer import MaterialMaix, LABELS, COLOR_IDS
from maix_target_ring import TargetRingMaix
from quality import Confirmation
from transport import LineBuffer, MaixUartTransport
from vision_protocol import parse_command, newer
from vision_runtime import VisionRuntime


def raw(class_id=4, x=300, y=200, w=40, h=40, score=0.9):
    return dict(class_id=class_id, x=x, y=y, w=w, h=h, score=score)


def material_result(objects=None):
    items = []
    for obj in ([raw()] if objects is None else objects):
        item = geometry(obj, 640, 480, 0.45, 6)
        if item is not None:
            label = LABELS[item["class_id"]]
            item.update(color=label, color_id=COLOR_IDS[label])
            items.append(item)
    flag_ambiguity(items)
    return {"mode": "MATERIAL", "image_size": {"width": 640, "height": 480}, "materials": items}


class Frame:
    def width(self):
        return 640

    def height(self):
        return 480


class FakeModel:
    def __init__(self, result, calls):
        self.result, self.calls = result, calls

    def detect(self, frame):
        self.calls.append(self.result["mode"])
        return copy.deepcopy(self.result)


def runtime():
    calls = []
    factories = {
        "MATERIAL": lambda **kw: FakeModel(material_result(), calls),
        "TARGET_RING": lambda **kw: FakeModel({
            "mode": "TARGET_RING", "image_size": {"width": 640, "height": 480},
            "targets": {"target_%d" % i: {"detected": False} for i in (1, 2, 3)}}, calls),
    }
    rt = VisionRuntime(factories=factories)
    # Tests use a neutral reference; the deployment config is calibrated separately.
    rt.reference.set(320, 240)
    return rt, calls


class CalibrationTests(unittest.TestCase):
    def test_offset_and_no_mutation(self):
        result = material_result()
        original = copy.deepcopy(result)
        out = PixelReference(320, 240).add_to_result(result)
        self.assertEqual(out["materials"][0]["offset"], {"x": 0, "y": -20})
        self.assertEqual(result, original)

    def test_invalid_reference_is_atomic(self):
        ref = PixelReference(320, 240)
        for x, y in [(640, 100), (-1, 0), (0, 480), (float("nan"), 0),
                     (0, float("inf")), (True, 0), (0, None)]:
            with self.assertRaises((ValueError, TypeError)):
                ref.set(x, y)
            self.assertEqual(ref.to_dict(), {"x": 320, "y": 240})


class DetectionTests(unittest.TestCase):
    def test_bad_geometry(self):
        for obj in [raw(w=0), raw(score=0.1), raw(x=float("nan")), raw(class_id=-1),
                    raw(class_id=9), raw(x=700), raw(score=float("inf"))]:
            self.assertIsNone(geometry(obj, 640, 480, 0.45, 6))

    def test_clipping_is_marked(self):
        item = geometry(raw(x=-20), 640, 480, 0.45, 6)
        self.assertTrue(item["touches_image_edge"])
        self.assertEqual(item["center"]["x"], 10)

    def test_duplicate_and_competing_labels(self):
        for objects in [[raw(), raw(x=400)], [raw(), raw(class_id=1)]]:
            result = material_result(objects)
            self.assertTrue(all(i["ambiguous"] for i in result["materials"]))

    def test_real_material_module_all_colors(self):
        detector = types.SimpleNamespace(detect=lambda *a, **k: [raw(class_id=i, x=20 + i*90) for i in range(6)])
        module = types.SimpleNamespace(image=types.SimpleNamespace(Fit=types.SimpleNamespace(FIT_CONTAIN=1)))
        with patch.dict(sys.modules, {"maix": module}), patch("material_recognizer.load_model", return_value=detector):
            result = MaterialMaix().detect(Frame())
        self.assertEqual([(i["color_id"], i["color"]) for i in result["materials"]],
                         [(1, "red"), (2, "yellow"), (3, "blue"), (4, "green"), (5, "black"), (6, "light_blue")])

    def test_real_ring_module_marks_same_number_ambiguous(self):
        detector = types.SimpleNamespace(detect=lambda *a, **k: [raw(class_id=0), raw(class_id=0, x=400)])
        module = types.SimpleNamespace(image=types.SimpleNamespace(Fit=types.SimpleNamespace(FIT_CONTAIN=1)))
        with patch.dict(sys.modules, {"maix": module}), patch("maix_target_ring.load_model", return_value=detector):
            result = TargetRingMaix().detect(Frame())
        self.assertTrue(result["targets"]["target_1"]["ambiguous"])
        self.assertFalse(result["targets"]["target_2"]["detected"])


class QualityTests(unittest.TestCase):
    def test_confirm_then_loss_never_reuses_center(self):
        quality = Confirmation()
        for frame in range(2):
            result = quality.apply(material_result(), frame * 100)
            self.assertEqual(result["materials"][0]["usable"], frame == 1)
        result = quality.apply(material_result([]), 300)
        self.assertEqual(result["status"], "NO_TARGET")
        self.assertEqual(result["materials"], [])
        self.assertFalse(quality.apply(material_result(), 400)["materials"][0]["usable"])

    def test_jump_long_gap_and_slow_drift(self):
        quality = Confirmation()
        for n in range(2):
            quality.apply(material_result(), n * 100)
        self.assertFalse(quality.apply(material_result([raw(x=400)]), 300)["materials"][0]["usable"])
        self.assertFalse(quality.apply(material_result([raw(x=400)]), 5000)["materials"][0]["usable"])
        quality.reset()
        # Small image motion is accepted; a larger frame-to-frame jump is not.
        for n in range(2):
            result = quality.apply(material_result([raw(x=300 + n * 4)]), n * 100)
            self.assertEqual(result["materials"][0]["usable"], n == 1)
        result = quality.apply(material_result([raw(x=320)]), 300)
        self.assertFalse(result["materials"][0]["usable"])

    def test_bad_candidates_never_confirm(self):
        for objects in [[raw(x=0)], [raw(w=2)], [raw(), raw(x=400)]]:
            quality = Confirmation()
            for frame in range(5):
                result = quality.apply(material_result(objects), frame * 100)
                self.assertFalse(any(i["usable"] for i in result["materials"]))


class RuntimeTests(unittest.TestCase):
    def test_all_stage_routes(self):
        rt, calls = runtime()
        modes = ["1", "2", "1", "2"]
        for seq, mode in enumerate(modes):
            rt.handle_command({"seq": seq, "command": "SET_MODE", "mode": mode})
            rt.detect(Frame(), seq * 100)
        self.assertEqual(calls, ["MATERIAL", "TARGET_RING", "MATERIAL", "TARGET_RING"])

    def test_reference_change_invalidates_old_result(self):
        rt, _ = runtime()
        rt.handle_command("1")
        old = rt.detect(Frame(), 100)
        rt.handle_command({"seq": 1, "command": "SET_REFERENCE", "x": 310, "y": 230})
        with self.assertRaises(ValueError):
            rt.packet(old, 120)
        new = rt.detect(Frame(), 200)
        self.assertEqual(new["materials"][0]["offset"], {"x": 10, "y": -10})
        self.assertFalse(new["materials"][0]["usable"])

    def test_invalid_command_does_not_change_state(self):
        rt, _ = runtime()
        rt.handle_command("1")
        for command in [{"seq": 0, "command": "SET_MODE", "mode": "9"},
                        {"seq": 0, "command": "SET_REFERENCE", "x": -1, "y": 0},
                        {"seq": 0, "command": "SET_MODE", "mode": "MISSING"}]:
            with self.assertRaises((TypeError, ValueError)):
                rt.handle_command(command)
            self.assertEqual(rt.mode, "1")
            self.assertIsNone(rt.last_command_seq)

    def test_sequence_duplicate_out_of_order_and_wrap(self):
        rt, _ = runtime()
        command = {"seq": 0xFFFFFFFF, "command": "SET_MODE", "mode": "2"}
        self.assertTrue(rt.handle_command(command))
        self.assertFalse(rt.handle_command(command))
        self.assertFalse(rt.handle_command(dict(command, seq=0xFFFFFFFE)))
        self.assertTrue(rt.handle_command(dict(command, seq=0)))

    def test_numeric_commands_select_two_requested_functions(self):
        rt, calls = runtime()
        for number, expected in (("1", "MATERIAL"), ("2", "TARGET_RING")):
            self.assertTrue(rt.handle_command(number + "\n"))
            self.assertEqual(rt.mode, number)
            result = rt.detect(Frame(), 100)
            self.assertEqual(result["mode"], expected)
        self.assertEqual(calls, ["MATERIAL", "TARGET_RING"])

    def test_load_failure_and_recovery(self):
        rt, _ = runtime()
        rt.detect(Frame(), 0)
        factory = rt.factories["TARGET_RING"]
        def fail(**kwargs):
            raise RuntimeError("NPU load failed")
        rt.factories["TARGET_RING"] = fail
        rt.handle_command({"seq": 1, "command": "SET_MODE", "mode": "2"})
        with self.assertRaises(RuntimeError):
            rt.detect(Frame(), 100)
        result = rt.empty_result("ERROR", 100)
        packet = json.loads(rt.packet(result, 110))
        self.assertEqual(packet, {"seq": 0, "items": []})
        self.assertIsNone(rt.recognizer)
        rt.factories["TARGET_RING"] = factory
        self.assertEqual(rt.detect(Frame(), 200)["mode"], "TARGET_RING")

    def test_packet_roundtrip_budget_and_sequence(self):
        rt, _ = runtime()
        rt.handle_command("1")
        for stamp in (100, 200, 300):
            result = rt.detect(Frame(), stamp)
        rt.encoder.sequence = 0xFFFFFFFF
        first = json.loads(rt.packet(result, 150))
        second = json.loads(rt.packet(result, 160))
        self.assertEqual((first["seq"], second["seq"]), (0xFFFFFFFF, 0))
        self.assertEqual(first, {"seq": 0xFFFFFFFF, "items": [[1, 0, -20]]})
        self.assertEqual(second["items"], [[1, 0, -20]])
        self.assertEqual(set(first), {"seq", "items"})
        result["materials"] = result["materials"] * config.MAX_OBJECTS
        self.assertLess(len(rt.packet(result, 160)), 1000)

    def test_idle_never_calls_model(self):
        rt, calls = runtime()
        rt.handle_command({"seq": 1, "command": "SET_MODE", "mode": "IDLE"})
        result = rt.detect(Frame(), 0)
        self.assertEqual(result["status"], "IDLE")
        self.assertEqual(calls, [])

    def test_stale_result_sends_no_coordinates(self):
        rt, _ = runtime()
        result = rt.detect(Frame(), 100)
        packet = json.loads(rt.packet(result, 100 + config.MAX_RESULT_AGE_MS + 1))
        self.assertEqual(packet["items"], [])

    def test_overflow_and_wrong_image_size(self):
        rt, _ = runtime()
        rt.handle_command("1")
        rt.factories["MATERIAL"] = lambda **k: FakeModel(material_result([raw()] * 7), [])
        self.assertEqual(rt.detect(Frame(), 0)["status"], "OVERFLOW")
        with self.assertRaises(ValueError):
            rt.detect(types.SimpleNamespace(width=lambda: 1280, height=lambda: 720), 100)


class FakeSerial:
    def __init__(self, data=b"", chunk=7):
        self.data, self.chunk, self.written = data, chunk, bytearray()

    def available(self, timeout):
        return len(self.data)

    def read(self, count, timeout=0):
        data, self.data = self.data[:count], self.data[count:]
        return data

    def write(self, data):
        count = min(self.chunk, len(data))
        self.written.extend(data[:count])
        return count

    def close(self):
        pass


class TransportTests(unittest.TestCase):
    def test_opens_maixcam_uart0_device(self):
        serial = FakeSerial()
        opened = []
        maix_module = types.SimpleNamespace(
            uart=types.SimpleNamespace(UART=lambda device, baudrate: opened.append((device, baudrate)) or serial))
        with patch.dict(sys.modules, {"maix": maix_module}):
            link = MaixUartTransport()
        self.assertEqual(opened, [("/dev/ttyS0", config.UART_BAUDRATE)])
        self.assertIs(link.serial, serial)

    def test_split_coalesced_and_crlf(self):
        buf = LineBuffer()
        self.assertEqual(buf.feed(b'{"seq":'), [])
        self.assertEqual(buf.feed(b'1}\r\nsecond\n'), ['{"seq":1}', 'second'])

    def test_oversize_noise_and_utf8_recover_at_newline(self):
        buf = LineBuffer(10)
        self.assertEqual(buf.feed(b'a' * 50), [])
        self.assertLessEqual(len(buf.buffer), 10)
        self.assertEqual(buf.feed(b'\n\xff\nvalid\n'), ['valid'])

    def test_short_writes_and_nonblocking_read(self):
        serial = FakeSerial(b'one\ntwo\n')
        link = MaixUartTransport(serial=serial)
        self.assertEqual(link.receive_commands(), ['one', 'two'])
        self.assertEqual(link.receive_commands(), [])
        link.send(b'{"seq":12}\n')
        self.assertEqual(serial.written, b'{"seq":12}\n')

    def test_failed_send_delimits_before_new_packet(self):
        serial = FakeSerial(chunk=0)
        link = MaixUartTransport(serial=serial)
        with self.assertRaises(OSError):
            link.send(b'partial\n')
        serial.chunk = 10
        link.send(b'new\n')
        self.assertEqual(serial.written, b'\nnew\n')

    def test_no_ack_on_command(self):
        import main
        serial = FakeSerial(b'{"seq":1,"command":"SET_MODE","mode":"IDLE"}\n')
        link = MaixUartTransport(serial=serial)
        rt, _ = runtime()
        main.poll_commands(link, rt)
        self.assertEqual(rt.mode, "IDLE")
        self.assertEqual(serial.written, b'')

    def test_parser_rejects_missing_sequence_and_unknown_command(self):
        for line in ['[]', '{}', 'SET_MODE IDLE', '{"seq":true,"command":"SET_MODE","mode":"IDLE"}',
                     '{"seq":-1,"command":"SET_MODE","mode":"IDLE"}', '{"seq":0,"command":"GET_MODE"}']:
            with self.assertRaises(ValueError):
                parse_command(line)


class MainLoopTests(unittest.TestCase):
    def test_command_arriving_during_inference_drops_old_result(self):
        import main
        rt, _ = runtime()
        rt.handle_command("1")
        frame = Frame()
        cam = types.SimpleNamespace(clear_buff=lambda: None, skip_frames=lambda n: None,
                                    read=lambda **kw: frame, close=lambda: None)
        serial = FakeSerial()
        link = MaixUartTransport(serial=serial)
        original_detect = rt.detect
        changed = False

        def detect(frame, stamp):
            nonlocal changed
            result = original_detect(frame, stamp)
            if not changed:
                serial.data = b'{"seq":1,"command":"SET_MODE","mode":"2"}\n'
                changed = True
            return result

        rt.detect = detect
        exits = iter([False, False, True])
        module = types.SimpleNamespace(
            app=types.SimpleNamespace(need_exit=lambda: next(exits)),
            camera=types.SimpleNamespace(Camera=lambda *a, **k: cam), display=None,
            image=types.SimpleNamespace(Format=types.SimpleNamespace(FMT_RGB888=1)))
        with patch.dict(sys.modules, {"maix": module}), patch.object(config, "SHOW_DISPLAY", False), \
                patch.object(config, "UART_ENABLED", True), \
                patch.object(main, "MaixUartTransport", return_value=link), \
                patch.object(main, "VisionRuntime", return_value=rt):
            main.main()
        packets = [json.loads(line) for line in bytes(serial.written).splitlines()]
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0], {"seq": 0, "items": []})


if __name__ == "__main__":
    unittest.main(verbosity=2)
