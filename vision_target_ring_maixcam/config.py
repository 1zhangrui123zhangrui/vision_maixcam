"""Deployment settings. Coordinates and tolerances are image pixels."""

WIDTH, HEIGHT = 640, 480
# 2026-10-07 photos 3/4/5 mean top-face center; assumes centered 16:9 -> 4:3 crop.
# Verify the magenta cross in the live 640x480 stream before robot motion.
MATERIAL_REFERENCE_X, MATERIAL_REFERENCE_Y = 347.4, 132.2
# Target-ring placement reference: the green center marker reported by the
# recognizer in the supplied live screenshot.
TARGET_REFERENCE_X, TARGET_REFERENCE_Y = 335.85, 181.15
REFERENCE_X, REFERENCE_Y = MATERIAL_REFERENCE_X, MATERIAL_REFERENCE_Y
# Wait for the controller's first command. The MCU selects 1 (material) or 2
# (target ring) over UART0; the camera never chooses a task by itself.
START_MODE = "IDLE"
UART_ENABLED = True
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200
SHOW_DISPLAY = True
SEND_INTERVAL_MS = 100
# A target center may be used even when the visible ring is incomplete.
CONFIDENCE = 0.35
IOU = 0.45
CONFIRM_FRAMES = 2
TARGET_CONFIRM_FRAMES = 2
MAX_CENTER_DRIFT = 8.0
TARGET_MAX_FRAME_STEP = 12.0
MAX_FRAME_GAP_MS = 900
MAX_RESULT_AGE_MS = 500
EDGE_MARGIN = 3
MIN_BOX_SIZE = 5
# A material touching the image border is accepted when most of its
# detector box is still visible. Only seriously clipped materials are held.
MATERIAL_MIN_VISIBLE_RATIO = 0.55
MATERIAL_MAX_FRAME_STEP = 8.0
MATERIAL_REJECT_EDGE = True
MAX_OBJECTS = 6
# Trial: bounded OpenCV ROI refinement. False selects YOLO box centers only.
RING_REFINE_ENABLED = True
RING_REFINE_INTERVAL = 8
RING_TRACK_MAX_AGE = 16
RING_TRACK_MAX_AGE_MS = 1800
RING_TRACK_MAX_SHIFT = 36.0
RING_TRACK_MAX_SIZE_CHANGE = 0.12
RING_REFINE_MAX_SIDE = 160
RING_REFINE_MAX_CONTOURS = 24
RING_REFINE_MAX_SAMPLES = 96
