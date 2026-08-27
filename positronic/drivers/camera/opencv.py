import sys
import time

import cv2

import pimm


class OpenCVCamera(pimm.ControlSystem):
    def __init__(self, camera_id: int, resolution: tuple[int, int], fps: int):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.frame = pimm.ControlSystemEmitter(self)
        self._frame_adapter = None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            raise RuntimeError(f'Failed to open camera {self.camera_id}')

        fps_counter = pimm.utils.RateCounter('OpenCV Camera')

        while not should_stop.value:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError('Failed to grab frame')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fps_counter.tick()
            # Use system time for timestamp since OpenCV doesn't provide frame timestamps
            self._frame_adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(frame, self._frame_adapter)
            self.frame.emit(self._frame_adapter)
            yield pimm.Yield()  # Give control back to the world


if __name__ == '__main__':
    from positronic.drivers.camera.video_writer import VideoWriter

    with pimm.World() as world:
        camera = OpenCVCamera(0, (640, 480), fps=30)
        writer = VideoWriter(sys.argv[1], 30)
        world.connect(camera.frame, writer.frame)

        for cmd in world.start(writer, camera):
            time.sleep(cmd.seconds if isinstance(cmd, pimm.Sleep) else 0)
