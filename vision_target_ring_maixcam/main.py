"""MaixCAM Pro: recognition, reference overlay and UART1 results."""

import sys
sys.dont_write_bytecode = True

import time
import config
from transport import MaixUartTransport, NullTransport
from vision_runtime import VisionRuntime, monotonic_ms


def poll_commands(link, runtime):
    try:
        lines = link.receive_commands()
    except Exception as exc:
        print("UART receive error:", exc)
        return
    for line in lines:
        try:
            runtime.handle_command(line)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            print("Ignored command:", exc)


def main():
    from maix import app, camera, display, image

    runtime = VisionRuntime()
    link = cam = disp = None
    next_send = 0
    try:
        link = MaixUartTransport() if config.UART_ENABLED else NullTransport()
        cam = camera.Camera(config.WIDTH, config.HEIGHT, image.Format.FMT_RGB888,
                            fps=30, buff_num=3)
        cam.skip_frames(5)
        disp = display.Display() if config.SHOW_DISPLAY else None
        while not app.need_exit():
            if config.UART_ENABLED:
                poll_commands(link, runtime)
            frame = None
            started = monotonic_ms()
            try:
                # Model loading occurs before capture, so load time does not age the image.
                runtime._load()
                cam.clear_buff()
                frame = cam.read(block=True, block_ms=1000)
                started = monotonic_ms()
                result = runtime.detect(frame, started)
            except Exception as exc:
                print("Vision error:", exc)
                result = runtime.empty_result("ERROR")
                time.sleep(0.1)
            # Commands may arrive during NPU inference. Never transmit the old
            # mode/reference result after applying such a command.
            if config.UART_ENABLED:
                poll_commands(link, runtime)
            if result["generation"] != runtime.generation:
                continue
            now = monotonic_ms()
            if now >= next_send:
                try:
                    link.send(runtime.packet(result, now))
                except Exception as exc:
                    print("UART send error:", exc)
                next_send = monotonic_ms() + config.SEND_INTERVAL_MS
            if disp is not None and frame is not None:
                disp.show(runtime.draw(frame, result))
    finally:
        for resource in (cam, disp, link):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print("Resource close error:", exc)


if __name__ == "__main__":
    main()
