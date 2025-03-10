import logging
from typing import Callable, List

import numpy as np

import ironic as ir
import geom
from positronic.tools.buttons import ButtonHandler

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler("teleop.log", mode="w")])


def front_position_franka_parser(teleop_transform: geom.Transform3D) -> geom.Transform3D:
    """
    This function maps the teleop transform to the franka robot frame, when operator is facing the robot
    from the front.

    It achieves this by permuting the axes xyz -> zxy.

    Args:
        teleop_transform: (geom.Transform3D) The input transform to be parsed.

    Returns:
        franka_transform: (geom.Transform3D) The transformed transform.
    """
    translation = teleop_transform.translation[[2, 0, 1]]
    # Apply rotation to map rotation around z -> y, x -> z, y -> x
    axis_permutation = geom.Rotation(0.5, 0.5, 0.5, 0.5)

    rotation = axis_permutation * teleop_transform.rotation * axis_permutation.inv
    return geom.Transform3D(translation, rotation)


def back_position_franka_parser(teleop_transform: geom.Transform3D) -> geom.Transform3D:
    """
    This function maps the teleop transform to the franka robot frame, when operator is facing the robot
    from the back.

    The same as front_position_franka_parser, but with additional X and Y axis flips.

    Args:
        teleop_transform: (geom.Transform3D) The input transform to be parsed.

    Returns:
        franka_transform: (geom.Transform3D) The transformed transform.
    """
    front_pos = front_position_franka_parser(teleop_transform)
    translation = front_pos.translation * [-1, -1, 1]

    flip_xy_rotation = geom.Rotation(0, 0, 0, 1)
    rotation = flip_xy_rotation * front_pos.rotation * flip_xy_rotation.inv

    return geom.Transform3D(translation, rotation)



@ir.ironic_system(
    input_ports=["teleop_transform", "teleop_buttons"],
    input_props=["robot_position"],
    output_ports=["robot_target_position", "gripper_target_grasp", "start_recording", "stop_recording", "reset"],
    output_props=["metadata"])
class TeleopSystem(ir.ControlSystem):

    def __init__(self, pos_parser: Callable[[geom.Transform3D], geom.Transform3D]):
        super().__init__()
        self.teleop_t = None
        self.offset = None
        self.is_tracking = False
        self.is_recording = False
        self.button_handler = ButtonHandler()
        self.pos_parser = pos_parser
        self.fps = ir.utils.FPSCounter("Teleop")

    def _parse_buttons(self, value: List[float]):
        if len(value) > 6:
            but = {'A': value[4], 'B': value[5], 'trigger': value[0], 'thumb': value[1], 'stick': value[3]}

            self.button_handler.update_buttons(but)

    @ir.out_property
    async def metadata(self):
        return ir.Message({'ui': 'teleop'})

    @ir.on_message("teleop_transform")
    async def handle_teleop_transform(self, message: ir.Message):
        self.teleop_t = self.pos_parser(message.data)
        self.fps.tick()

        if self.is_tracking and self.offset is not None:
            target = geom.Transform3D(self.teleop_t.translation + self.offset.translation,
                                 self.teleop_t.rotation * self.offset.rotation)
            await self.outs.robot_target_position.write(ir.Message(target, message.timestamp))

    @ir.on_message("teleop_buttons")
    async def handle_teleop_buttons(self, message: ir.Message):
        self._parse_buttons(message.data)

        track_but = self.button_handler.is_pressed('A')
        record_but = self.button_handler.is_pressed('B')
        reset_but = self.button_handler.is_pressed('stick')

        grasp_but = self.button_handler.get_value('trigger')

        if self.is_tracking:
            await self.outs.gripper_target_grasp.write(ir.Message(grasp_but, message.timestamp))

        if track_but:
            await self._switch_tracking(message.timestamp)

        if record_but:
            await self._switch_recording(message.timestamp)

        if reset_but:
            await self.outs.reset.write(ir.Message(None, message.timestamp))
            if self.is_tracking:
                await self._switch_tracking(message.timestamp)
            if self.is_recording:
                await self._switch_recording(message.timestamp)

    async def _switch_tracking(self, timestamp: int):
        # Note that translation and rotation offsets are independent
        if self.teleop_t is not None:
            robot_t = (await self.ins.robot_position()).data
            self.offset = geom.Transform3D(-self.teleop_t.translation + robot_t.translation,
                                      self.teleop_t.rotation.inv * robot_t.rotation)
        if self.is_tracking:
            logging.info('Stopped tracking')
            self.is_tracking = False
        else:
            logging.info('Started tracking')
            self.is_tracking = True

    async def _switch_recording(self, timestamp: int):
        if self.is_recording:
            logging.info('Stopped recording')
            self.is_recording = False
            await self.outs.stop_recording.write(ir.Message(None, timestamp))
        else:
            logging.info('Started recording')
            self.is_recording = True
            await self.outs.start_recording.write(ir.Message(None, timestamp))
