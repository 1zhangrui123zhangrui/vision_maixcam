"""Bounded nonblocking command input and complete, unacknowledged UART writes."""

import time
import config


class LineBuffer:
    def __init__(self, limit=256):
        self.limit = limit
        self.buffer = bytearray()
        self.discard = False

    def feed(self, data):
        lines = []
        for byte in data:
            if byte == 10:
                if not self.discard and self.buffer:
                    try:
                        lines.append(bytes(self.buffer).decode("ascii").strip())
                    except UnicodeDecodeError:
                        pass
                self.buffer.clear()
                self.discard = False
            elif not self.discard:
                if len(self.buffer) >= self.limit:
                    self.buffer.clear()
                    self.discard = True
                else:
                    self.buffer.append(byte)
        return lines


class NullTransport:
    def receive_commands(self):
        return []

    def send(self, packet):
        pass

    def close(self):
        pass


class MaixUartTransport:
    def __init__(self, device=config.UART_DEVICE, baudrate=config.UART_BAUDRATE, serial=None):
        if serial is None:
            from maix import uart

            if device != "/dev/ttyS0":
                raise ValueError("MaixCAM Pro communication uses UART0 (/dev/ttyS0)")
            serial = uart.UART(device, baudrate)
        self.serial = serial
        self.lines = LineBuffer()
        self.needs_delimiter = False

    def receive_commands(self):
        available = self.serial.available(0)
        if available < 0:
            raise OSError("UART receive failed")
        if not available:
            return []
        data = self.serial.read(min(available, 1024), timeout=0)
        return self.lines.feed(data or b"")

    def send(self, packet):
        # A short write is not an ACK failure: finish only the unwritten suffix.
        # After a failed write, terminate the partial line before the next packet.
        data = (b"\n" if self.needs_delimiter else b"") + packet
        offset = 0
        deadline = time.monotonic() + 0.2
        try:
            while offset < len(data):
                count = self.serial.write(data[offset:])
                if not isinstance(count, int) or not 0 < count <= len(data) - offset:
                    raise OSError("UART write made no progress")
                offset += count
                if offset < len(data) and time.monotonic() >= deadline:
                    raise OSError("UART write timed out")
        except Exception:
            self.needs_delimiter = True
            raise
        self.needs_delimiter = False

    def close(self):
        self.serial.close()
