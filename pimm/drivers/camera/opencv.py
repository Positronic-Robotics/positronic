from typing import Tuple

import cv2

import ironic2 as ir
from ironic.utils import FPSCounter


class OpenCVCamera:

    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, camera_id: int, resolution: Tuple[int, int], fps: int):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

    def run(self, should_stop: ir.SignalReader):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        fps_counter = FPSCounter('OpenCV Camera')

        while not ir.is_true(should_stop):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to grab frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fps_counter.tick()
            # Use system time for timestamp since OpenCV doesn't provide frame timestamps
            self.frame.emit({'frame': frame})


if __name__ == "__main__":
    import sys
    import time
    import av

    # We could implement this as a plain function
    # TODO: Extract this into utilities
    class VideoWriter:
        frame: ir.SignalReader = ir.NoOpReader()

        def __init__(self, filename: str, fps: int, codec: str = 'libx264'):
            self.filename = filename
            self.fps = fps
            self.codec = codec

        def run(self, should_stop: ir.SignalReader):
            container = av.open(self.filename, mode='w', format='mp4')
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}

            last_ts = None
            while not ir.is_true(should_stop):
                message = self.frame.value()
                if message is ir.NoValue or last_ts == message.ts:
                    time.sleep(0.5 / self.fps)
                    continue
                last_ts = message.ts

                frame = av.VideoFrame.from_ndarray(message.data['frame'], format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)

            container.close()

    with ir.World() as world:
        camera = OpenCVCamera(0, (640, 480), fps=30)
        writer = VideoWriter(sys.argv[1], 30)

        camera.frame, writer.frame = world.pipe()
        world.start(camera.run)
        writer.run()
