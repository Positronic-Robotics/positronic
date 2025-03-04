from typing import Dict, Tuple
import cv2
import numpy as np
import ironic as ir


class OpenCVCamera:
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.fps_counter = ir.utils.FPSCounter('OpenCVCamera')
        self.cap = None

    def setup(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

    def get_frame(self) -> Tuple[Dict[str, np.ndarray], int]:
        assert self.cap is not None, "You must call setup() before get_frame()"

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use system time for timestamp since OpenCV doesn't provide frame timestamps
        timestamp_ns = ir.system_clock()
        self.fps_counter.tick()
        return {'image': frame}, timestamp_ns

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()


@ir.ironic_system(output_ports=['frame'])
class OpenCVCameraCS(ir.ControlSystem):
    def __init__(self, camera: OpenCVCamera):
        super().__init__()
        self.camera = camera

    async def setup(self):
        self.camera.setup()

    async def step(self):
        frame, timestamp = self.camera.get_frame()
        await self.outs.frame.write(ir.Message(data=frame, timestamp=timestamp))
        return ir.State.ALIVE

    async def cleanup(self):
        self.camera.cleanup()
