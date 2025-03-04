from typing import Dict, Tuple

import numpy as np
import pyrealsense2 as rs

import ironic as ir


class RealsenseCamera:
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
        enable_infrared: bool = True,
    ):
        self.resolution = resolution
        self.fps = fps
        self.config = rs.config()
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared


    def setup(self):
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.rgb8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
        if self.enable_infrared:
            self.config.enable_stream(rs.stream.infrared, 1, self.resolution[0], self.resolution[1], rs.format.y8, self.fps)
            self.config.enable_stream(rs.stream.infrared, 2, self.resolution[0], self.resolution[1], rs.format.y8, self.fps)

        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

    def get_frame(self) -> Tuple[Dict[str, np.ndarray], float]:
        assert self.pipeline is not None, "You must call setup() before get_frame()"

        realsense_frames = self.pipeline.wait_for_frames()

        frames = {}

        if self.enable_color:
            color_frame = realsense_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            frames['image'] = color_frame
        if self.enable_depth:
            depth_frame = realsense_frames.get_depth_frame()
            depth_frame = self.get_metric_depth(depth_frame)
            frames['depth'] = depth_frame
        if self.enable_infrared:
            infrared_frame_1 = realsense_frames.get_infrared_frame(1)
            infrared_frame_1 = np.asanyarray(infrared_frame_1.get_data())
            frames['infrared_1'] = infrared_frame_1
            infrared_frame_2 = realsense_frames.get_infrared_frame(2)
            infrared_frame_2 = np.asanyarray(infrared_frame_2.get_data())
            frames['infrared_2'] = infrared_frame_2

        timestamp_ms = realsense_frames.get_timestamp()
        timestamp_ns = timestamp_ms * 1e6

        return frames, int(timestamp_ns)

    def get_metric_depth(self, depth_frame) -> np.ndarray:
        depth_units = depth_frame.get_units()
        depth_frame = np.asanyarray(depth_frame.get_data())
        depth_frame = depth_frame.astype(np.float32)
        depth_frame *= depth_units
        return depth_frame


@ir.ironic_system(output_ports=['frame'])
class RealsenseCameraCS(ir.ControlSystem):
    def __init__(self, camera: RealsenseCamera):
        super().__init__()
        self.camera = camera

    async def setup(self):
        self.camera.setup()

    async def step(self):
        frame, timestamp = self.camera.get_frame()
        await self.outs.frame.write(ir.Message(data=frame, timestamp=timestamp))
        return ir.State.ALIVE
