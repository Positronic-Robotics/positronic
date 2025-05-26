from functools import partial
from typing import Tuple

import cv2

import ironic2 as ir
from ironic import system_clock
from ironic.utils import FPSCounter


def _opencv_internal(camera_id: int, resolution: Tuple[int, int], fps: int, stopped, frames: ir.Channel):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")

    fps_counter = FPSCounter('OpenCV Camera')

    while not stopped.is_set():
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fps_counter.tick()
        # Use system time for timestamp since OpenCV doesn't provide frame timestamps
        frames.write(ir.Message(data={'frame': frame}, timestamp=system_clock()))


def opencv_camera(camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
    return partial(_opencv_internal, camera_id, resolution, fps)


if __name__ == "__main__":
    import sys
    import time
    import av

    def main_loop(stopped, frames):
        codec, fps, filename = 'libx264', 30, sys.argv[1]

        container = av.open(filename, mode='w', format='mp4')
        stream = container.add_stream(codec, rate=fps)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}
        while not stopped.is_set():
            message = frames.read()
            if message is ir.NoValue:
                time.sleep(0.03)
                continue

            if stream.width is None:  # First frame
                stream.width = message.data['frame'].shape[1]
                stream.height = message.data['frame'].shape[0]

            frame = av.VideoFrame.from_ndarray(message.data['frame'], format='rgb24')
            packet = stream.encode(frame)
            container.mux(packet)
        container.close()

    frames = ir.mp.QueueChannel()
    world = ir.mp.MPWorld()
    world.add_background_loop(opencv_camera(0, (640, 480), fps=30), frames)
    world.run(main_loop, frames)
