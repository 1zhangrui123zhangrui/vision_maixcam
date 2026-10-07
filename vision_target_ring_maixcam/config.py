"""Deployment settings. Coordinates and tolerances are image pixels."""

WIDTH, HEIGHT = 640, 480
# 2026-10-07 photos 3/4/5 mean top-face center; assumes centered 16:9 -> 4:3 crop.
# Verify the magenta cross in the live 640x480 stream before robot motion.
MATERIAL_REFERENCE_X, MATERIAL_REFERENCE_Y = 347.4, 132.2
# Target-ring placement reference: the green center marker reported by the
# recognizer in the supplied live screenshot.
TARGET_REFERENCE_X, TARGET_REFERENCE_Y = 355.0, 178.0
REFERENCE_X, REFERENCE_Y = MATERIAL_REFERENCE_X, MATERIAL_REFERENCE_Y
# Wait for the controller's first command.  The MCU selects 1 (material) or
# 2 (target ring) over UART0; the camera never chooses a task by itself.
START_MODE = "IDLE"
UART_ENABLED = True
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200
SHOW_DISPLAY = True
SEND_INTERVAL_MS = 100
# A target center may be used even when the visible ring is incomplete.
CONFIDENCE = 0.35
IOU = 0.45
CONFIRM_FRAMES = 3
MAX_CENTER_DRIFT = 8.0
MAX_FRAME_GAP_MS = 900
MAX_RESULT_AGE_MS = 500
EDGE_MARGIN = 3
MIN_BOX_SIZE = 5
MAX_OBJECTS = 6
# Device-side ring-center estimator.  It is intentionally small enough for
# MaixCAM; the PC implementation remains the higher-precision validator.
RING_ROI_PADDING = 0.08
RING_DARK_THRESHOLD = 105
RING_MIN_PIXELS = 12
RING_CENTER_MAX_SHIFT_RATIO = 0.24
