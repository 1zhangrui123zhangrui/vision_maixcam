"""Exercise the deployed estimator with known synthetic image-space centers."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import types
import json
import copy
from contextlib import contextmanager

import cv2
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision_target_ring_maixcam"))
import config
from ring_center import refine_center
from maix_target_ring import TargetRingMaix
from quality import Confirmation
from calibration import PixelReference
from vision_protocol import ResultPacketEncoder
from test_device_runtime import material_result


def sample(digit="1", angle=0, axes=(135, 100)):
    image = np.full((480, 640, 3), 225, np.uint8)
    for factor in (.58, .66, .83, 1):
        cv2.ellipse(image, (355, 220), tuple(round(v * factor) for v in axes),
                    angle, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, digit, (322, 248), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 8)
    return image


BOX = {"x": 214, "y": 115, "w": 298, "h": 220}


@contextmanager
def tracked_device():
    state = {"objects": [dict(BOX, score=.9, class_id=0)], "now": 0.0}
    frame = types.SimpleNamespace(width=lambda: 640, height=lambda: 480)
    detector = types.SimpleNamespace(detect=lambda *a, **k: copy.deepcopy(state["objects"]))
    image = types.SimpleNamespace(Fit=types.SimpleNamespace(FIT_CONTAIN=1),
                                  image2cv=lambda *a, **kw: None)
    with patch.dict(sys.modules, {"maix": types.SimpleNamespace(image=image)}), \
            patch("maix_target_ring.load_model", return_value=detector), \
            patch("maix_target_ring.time.monotonic", side_effect=lambda: state["now"]):
        recognizer = TargetRingMaix()
        recognizer.refine = lambda *args: dict(center={"x": 355., "y": 220.},
            center_source="multi_ellipse", center_quality=.9, ring_count=3, center_usable=True)
        yield recognizer, frame, state


class CenterTests(unittest.TestCase):
    def test_deployment_resolution_accuracy(self):
        for digit in "123":
            for angle in (0, 20, -25):
                out = refine_center(sample(digit, angle), BOX, config.RING_REFINE_MAX_SIDE,
                                    config.RING_REFINE_MAX_CONTOURS, config.RING_REFINE_MAX_SAMPLES)
                self.assertIsNotNone(out)
                self.assertLess(np.hypot(out["center"]["x"] - 355, out["center"]["y"] - 220), 2)

    def test_continuous_motion_jump_loss_and_ambiguity(self):
        quality = Confirmation()
        def moving(x, ambiguous=False):
            item = material_result()["materials"][0]
            item.update(class_id=0, target_id=1, detected=True, ambiguous=ambiguous,
                        center={"x": x, "y": 178.}, center_source="tracked")
            item.pop("color_id")
            return {"mode": "TARGET_RING", "targets": {"target_1": item}}
        for n in range(30):
            out = quality.apply(moving(200 + 5 * n), n * 100)
            self.assertEqual(out["targets"]["target_1"]["usable"], n >= 1)
            out = PixelReference().add_to_result(out)
            out["timestamp_ms"] = n * 100
            packet = json.loads(ResultPacketEncoder().encode(out, n * 100))
            self.assertEqual(packet["items"], [[1,
                round(200 + 5*n - config.TARGET_REFERENCE_X, 2),
                round(178 - config.TARGET_REFERENCE_Y, 2)]] if n else [])
        self.assertFalse(quality.apply(moving(500), 3000)["targets"]["target_1"]["usable"])
        self.assertTrue(quality.apply(moving(501), 3100)["targets"]["target_1"]["usable"])
        quality.apply({"mode": "TARGET_RING", "targets": {}}, 3200)
        self.assertFalse(quality.apply(moving(502), 3300)["targets"]["target_1"]["usable"])
        for n in range(3):
            self.assertFalse(quality.apply(moving(502, True), 3400+n*100)["targets"]["target_1"]["usable"])
        self.assertFalse(quality.apply(moving(502), 5000)["targets"]["target_1"]["usable"])

    def test_failed_refinement_does_not_refresh_cache_or_retry_each_frame(self):
        with tracked_device() as (recognizer, frame, state):
            recognizer.detect(frame)
            refine = unittest.mock.Mock(return_value=None)
            recognizer.refine = refine
            for n in range(2, 19):
                state["now"] = n * .05
                out = recognizer.detect(frame)["targets"]["target_1"]
            self.assertEqual(refine.call_count, 2)
            self.assertEqual(out["center_source"], "bbox_maix")
            self.assertEqual(recognizer.tracks, {})

    def test_tracking_invalidation(self):
        for reason in ("loss", "time", "size", "shift", "edge", "ambiguous"):
            with self.subTest(reason=reason), tracked_device() as (recognizer, frame, state):
                recognizer.detect(frame)
                recognizer.refine = lambda *args: None
                if reason == "loss":
                    saved = state["objects"]
                    state["objects"] = []
                    self.assertFalse(recognizer.detect(frame)["targets"]["target_1"]["detected"])
                    state["objects"] = saved
                elif reason == "time":
                    state["now"] = 1.801
                elif reason == "size":
                    state["objects"][0]["w"] *= 1.2
                elif reason == "shift":
                    state["objects"][0]["x"] += 40
                elif reason == "edge":
                    recognizer.tracks[0]["edge"] = True
                elif reason == "ambiguous":
                    state["objects"] *= 2
                out = recognizer.detect(frame)["targets"]["target_1"]
                self.assertEqual(out["center_source"], "bbox_maix")
                self.assertEqual(recognizer.tracks, {})

    def test_one_fit_per_frame_with_fair_failed_retries(self):
        with tracked_device() as (recognizer, frame, state):
            state["objects"] = [dict(x=40+180*i, y=100, w=120, h=100, score=.9, class_id=i)
                                for i in range(3)]
            boxes = []
            recognizer.refine = lambda rgb, box, *args: boxes.append(box["x"])
            counts = []
            for n in range(11):
                recognizer.detect(frame)
                counts.append(len(boxes))
            self.assertEqual(boxes, [40, 220, 400, 40, 220, 400])
            self.assertTrue(all(b-a <= 1 for a, b in zip([0]+counts, counts)))

    def test_digits_offset_box_and_rotation(self):
        for digit in "123":
            for angle in (0, 20, -25):
                with self.subTest(digit=digit, angle=angle):
                    result = refine_center(sample(digit, angle), BOX)
                    self.assertIsNotNone(result)
                    self.assertLess(np.hypot(result["center"]["x"] - 355,
                                             result["center"]["y"] - 220), 2)

    def test_partial_outer_ring(self):
        image = sample()
        image[:, 472:] = 225
        result = refine_center(image, BOX)
        self.assertIsNotNone(result)
        self.assertLess(np.hypot(result["center"]["x"] - 355, result["center"]["y"] - 220), 2)

    def test_flat_ellipse_and_single_ring_rejection(self):
        result = refine_center(sample(axes=(135, 55)), dict(x=214, y=155, w=298, h=140))
        self.assertIsNotNone(result)
        self.assertLess(np.hypot(result["center"]["x"] - 355, result["center"]["y"] - 220), 2)
        image = np.full((480, 640, 3), 225, np.uint8)
        cv2.ellipse(image, (355, 220), (135, 100), 0, 0, 360, (20, 20, 20), 3)
        self.assertIsNone(refine_center(image, BOX))

    def test_noise_and_blur(self):
        rng = np.random.default_rng(12)
        image = np.clip(sample().astype(float) + rng.normal(0, 7, (480, 640, 3)), 0, 255).astype(np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 1)
        result = refine_center(image, BOX)
        self.assertIsNotNone(result)
        self.assertLess(np.hypot(result["center"]["x"] - 355, result["center"]["y"] - 220), 2)

    def test_blank_digit_only_and_small_roi_reject(self):
        image = np.full((480, 640, 3), 225, np.uint8)
        self.assertIsNone(refine_center(image, BOX))
        cv2.putText(image, "3", (320, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        self.assertIsNone(refine_center(image, BOX))
        self.assertIsNone(refine_center(image, dict(x=300, y=200, w=20, h=20)))

    def test_input_is_not_modified_and_border_coordinates(self):
        image = sample()
        original = image.copy()
        refine_center(image, BOX)
        np.testing.assert_array_equal(image, original)
        cropped = image[:, 240:]
        result = refine_center(cropped, dict(x=0, y=115, w=272, h=220))
        self.assertIsNotNone(result)
        self.assertLess(np.hypot(result["center"]["x"] - 115, result["center"]["y"] - 220), 2)

    def test_device_adapter_rgb_refinement_and_switch(self):
        pixels = sample()
        frame = types.SimpleNamespace(width=lambda: 640, height=lambda: 480)
        detector = types.SimpleNamespace(detect=lambda *a, **k: [dict(BOX, score=.9, class_id=0)])
        image = types.SimpleNamespace(Fit=types.SimpleNamespace(FIT_CONTAIN=1),
                                      image2cv=lambda *a, **kw: pixels)
        with patch.dict(sys.modules, {"maix": types.SimpleNamespace(image=image)}), \
                patch("maix_target_ring.load_model", return_value=detector):
            result = TargetRingMaix().detect(frame)
            self.assertEqual(result["targets"]["target_1"]["center_source"], "multi_ellipse")
            with patch.object(config, "RING_REFINE_ENABLED", False):
                plain = TargetRingMaix().detect(frame)
            self.assertEqual(plain["targets"]["target_1"]["center_source"], "bbox_maix")
            image.image2cv = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("conversion unavailable"))
            fallback = TargetRingMaix().detect(frame)
            self.assertEqual(fallback["targets"]["target_1"]["center_source"], "bbox_maix")

    def test_source_change_keeps_moving_target_track_and_packet_reference(self):
        quality = Confirmation()
        def result(source):
            item = material_result()["materials"][0]
            item.update(class_id=0, target_id=1, detected=True, center_source=source,
                        center={"x": 355., "y": 178.}, touches_image_edge=True)
            item.pop("color_id")
            return {"mode": "TARGET_RING", "targets": {"target_1": item}}
        for n in range(3):
            out = quality.apply(result("bbox_maix"), n * 100)
        self.assertTrue(out["targets"]["target_1"]["usable"])
        for n in range(2):
            out = quality.apply(result("multi_ellipse"), 300 + n * 100)
            self.assertTrue(out["targets"]["target_1"]["usable"])
        out = PixelReference().add_to_result(out)
        out["timestamp_ms"] = 500
        self.assertEqual(json.loads(ResultPacketEncoder().encode(out, 500))["items"], [[1,
            round(355 - config.TARGET_REFERENCE_X, 2),
            round(178 - config.TARGET_REFERENCE_Y, 2)]])
        self.assertEqual(json.loads(ResultPacketEncoder().encode(out, 1001))["items"], [])

    def test_incomplete_ring_box_center_can_send_after_two_frames(self):
        quality = Confirmation()
        item = material_result()["materials"][0]
        item.update(class_id=0, target_id=1, detected=True, center_source="bbox_maix",
                    center={"x": 330., "y": 180.}, touches_image_edge=True)
        item.pop("color_id")
        for n in range(2):
            result = quality.apply({"mode": "TARGET_RING",
                                    "targets": {"target_1": copy.deepcopy(item)}}, n * 100)
            self.assertEqual(result["targets"]["target_1"]["usable"], n == 1)

    def test_device_tracks_precise_center_between_geometry_passes(self):
        pixels = sample()
        frame = types.SimpleNamespace(width=lambda: 640, height=lambda: 480)
        detections = [dict(BOX, score=.9, class_id=0), dict(BOX, x=219, score=.9, class_id=0)]
        detector = types.SimpleNamespace(detect=lambda *a, **k: [detections.pop(0)])
        image = types.SimpleNamespace(Fit=types.SimpleNamespace(FIT_CONTAIN=1),
                                      image2cv=lambda *a, **kw: pixels)
        with patch.dict(sys.modules, {"maix": types.SimpleNamespace(image=image)}), \
                patch("maix_target_ring.load_model", return_value=detector), \
                patch.object(config, "RING_REFINE_INTERVAL", 8):
            recognizer = TargetRingMaix()
            first = recognizer.detect(frame)["targets"]["target_1"]
            second = recognizer.detect(frame)["targets"]["target_1"]
        self.assertEqual(first["center_source"], "multi_ellipse")
        self.assertEqual(second["center_source"], "tracked")
        self.assertAlmostEqual(second["center"]["x"] - first["center"]["x"], 5, delta=.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
