import time
from typing import Tuple

import cv2

import pimm


class OpenCVCamera(pimm.ControlSystem):
    def __init__(self, camera_id: int, resolution: Tuple[int, int], fps: int):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.frame = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        fps_counter = pimm.utils.RateCounter('OpenCV Camera')

        while not should_stop.value:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to grab frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fps_counter.tick()
            # Use system time for timestamp since OpenCV doesn't provide frame timestamps
            self.frame.emit({'image': frame})
            yield pimm.Pass()  # Give control back to the world


if __name__ == "__main__":
    import sys
    import av

    # We could implement this as a plain function
    # TODO: Extract this into utilities
    class VideoWriter(pimm.ControlSystem):

        def __init__(self, filename: str, fps: int, codec: str = 'libx264'):
            self.filename = filename
            self.fps = fps
            self.codec = codec
            self.frame = pimm.ControlSystemReceiver(self)

        def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
            print(f"Writing to {self.filename}")
            fps_counter = pimm.utils.RateCounter('VideoWriter')
            with av.open(self.filename, mode='w', format='mp4') as container:
                stream = container.add_stream(self.codec, rate=self.fps)
                stream.pix_fmt = 'yuv420p'
                stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}

                frame_receiver = pimm.DefaultReceiver(pimm.ValueUpdated(self.frame), (None, False))
                while not should_stop.value:
                    frame, updated = frame_receiver.value
                    if not updated:
                        yield pimm.Sleep(0.5 / self.fps)
                        continue

                    frame = av.VideoFrame.from_ndarray(frame['image'], format='rgb24')
                    packet = stream.encode(frame)
                    container.mux(packet)
                    fps_counter.tick()

    with pimm.World() as world:
        camera = OpenCVCamera(0, (640, 480), fps=30)
        writer = VideoWriter(sys.argv[1], 30)
        world.connect(camera.frame, writer.frame)

        for sleep_cmd in world.start(writer, camera):
            time.sleep(sleep_cmd.seconds)
