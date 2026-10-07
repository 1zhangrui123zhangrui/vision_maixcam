"""Deployment settings. Coordinates and tolerances are image pixels."""

WIDTH, HEIGHT = 640, 480
# 2026-10-07 photos 3/4/5 mean top-face center; assumes centered 16:9 -> 4:3 crop.
# Verify the magenta cross in the live 640x480 stream before robot motion.
REFERENCE_X, REFERENCE_Y = 347.4, 132.2
START_MODE = "IDLE"
UART_ENABLED = True
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200
SHOW_DISPLAY = True
SEND_INTERVAL_MS = 100
CONFIDENCE = 0.45
IOU = 0.45
CONFIRM_FRAMES = 3
MAX_CENTER_DRIFT = 6.0
MAX_FRAME_GAP_MS = 500
MAX_RESULT_AGE_MS = 500
EDGE_MARGIN = 3
MIN_BOX_SIZE = 8
MAX_OBJECTS = 6
