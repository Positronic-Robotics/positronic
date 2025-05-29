from typing import Tuple

import cv2

import ironic2 as ir
from ironic.utils import FPSCounter


class OpenCVCamera(ir.ControlSystem):

    def __init__(self, comms: ir.CommunicationProvider, camera_id: int, resolution: Tuple[int, int], fps: int):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        self.frame = comms.emitter('frame')
        self.should_stop = comms.should_stop()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        fps_counter = FPSCounter('OpenCV Camera')

        while not ir.signal_is_true(self.should_stop):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to grab frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fps_counter.tick()
            # Use system time for timestamp since OpenCV doesn't provide frame timestamps
            self.frame.emit(ir.Message(data={'frame': frame}, ts=ir.system_clock()))


if __name__ == "__main__":
    import sys
    import time
    import av

    world = ir.mp.MPWorld()
    camera_interface = world.add_background_control_system(OpenCVCamera, 0, (640, 480), fps=30)

    def main_loop(should_stop: ir.SignalReader):
        codec, fps, filename = 'libx264', 30, sys.argv[1]

        container = av.open(filename, mode='w', format='mp4')
        stream = container.add_stream(codec, rate=fps)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}
        last_ts = None
        while not ir.signal_is_true(should_stop):
            message = camera_interface.frame.value()
            if message is ir.NoValue or last_ts == message.ts:
                time.sleep(0.03)
                continue

            last_ts = message.ts
            if stream.width is None:  # First frame
                stream.width = message.data['frame'].shape[1]
                stream.height = message.data['frame'].shape[0]

            frame = av.VideoFrame.from_ndarray(message.data['frame'], format='rgb24')
            packet = stream.encode(frame)
            container.mux(packet)
        container.close()

    world.run(main_loop)
