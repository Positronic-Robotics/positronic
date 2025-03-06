import numpy as np
import rerun as rr
from typing import Dict, Tuple

import ironic as ir
import cv2


def _wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def _depth_image_to_uint8(depth_image: np.ndarray, depth_range_m: Tuple[float, float]) -> np.ndarray:
    depth_image = depth_image.astype(np.float32)
    depth_image = (depth_image - depth_range_m[0]) / (depth_range_m[1] - depth_range_m[0])
    depth_image = np.clip(depth_image, 0.0, 1.0)
    depth_image = (depth_image * 255).astype(np.uint8)
    return depth_image


def _add_crosshair_to_image(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    # add vertical and horizontal lines to the image
    height, width = image.shape[:2]
    line_thickness = 2
    line_color = (255, 0, 0)
    cv2.line(image, (width // 2, 0), (width // 2, height), line_color, line_thickness)
    cv2.line(image, (0, height // 2), (width, height // 2), line_color, line_thickness)

    # add horizontal line 10% from the bottom of the image
    sponge_color = (0, 255, 255)
    cv2.line(image, (0, height - height // 10), (width, height - height // 10), sponge_color, line_thickness)
    return image


@ir.ironic_system(
    input_ports=['frame', 'new_recording'],
    input_props=['ext_force_ee', 'ext_force_base', 'robot_position'],
)
class RerunVisualiser(ir.ControlSystem):
    """Control system for visualizing data streams using rerun."""

    def __init__(self, port: int = 9876, depth_image_range_m: Tuple[float, float] = (0.15, 3.0)) -> None:
        super().__init__()
        self.depth_image_range_m = depth_image_range_m
        self.recording_id = 0
        self.port = port

    async def setup(self):
        """Initialize rerun recording."""
        self.application_id = "Data Collection1"
        rr.init(self.application_id, recording_id=self.recording_id)
        rr.spawn(memory_limit="50%", port=self.port)
        await self.on_new_recording(ir.Message(data=None))

    @ir.on_message('new_recording')
    async def on_new_recording(self, msg: ir.Message):
        """Handle new recording message."""
        self.recording_id += 1
        rr.new_recording(self.application_id, recording_id=self.recording_id, make_default=True)
        rr.connect()

    @ir.on_message('frame')
    async def on_frame(self, msg: ir.Message):
        """Handle incoming camera frames."""
        frames: Dict[str, np.ndarray] = msg.data
        rr.set_time_nanos("camera", nanos=msg.timestamp)
        print(frames.keys())

        # Handle RGB image
        for key, image in frames.items():
            if 'image' in key:
                image = _add_crosshair_to_image(image)
            rr.log(f"camera/{key}", rr.Image(image).compress())

        if 'depth' in frames:
            depth_m = frames['depth']
            depth_m = _depth_image_to_uint8(depth_m, self.depth_image_range_m)
            rr.log("camera/depth", rr.Image(depth_m).compress())

        if 'infrared_1' in frames:
            rr.log("camera/infrared_1", rr.Image(frames['infrared_1']).compress())
        if 'infrared_2' in frames:
            rr.log("camera/infrared_2", rr.Image(frames['infrared_2']).compress())

        if self.is_bound('ext_force_ee'):
            ee_force = np.array((await self.ins.ext_force_ee()).data)
            rr.log("forces/end_effector/x", rr.Scalar(ee_force[0]))
            rr.log("forces/end_effector/y", rr.Scalar(ee_force[1]))
            rr.log("forces/end_effector/z", rr.Scalar(ee_force[2]))
            rr.log("robot/ee_force", rr.Arrows3D(origins=robot_position.translation.copy(), vectors=ee_force[:3].copy()))

        if self.is_bound('ext_force_base'):

            base_force = (await self.ins.ext_force_base()).data
            rr.log("forces/base/x", rr.Scalar(base_force[0]))
            rr.log("forces/base/y", rr.Scalar(base_force[1]))
            rr.log("forces/base/z", rr.Scalar(base_force[2]))

        if self.is_bound('robot_position'):
            robot_position = (await self.ins.robot_position()).data
            rr_robot_position = rr.Transform3D(translation=robot_position.translation.copy(),
                                               rotation=rr.Quaternion(xyzw=_wxyz_to_xyzw(robot_position.rotation.as_quat)))
            rr.log("robot/position", rr_robot_position)
