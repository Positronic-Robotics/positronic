import logging
from typing import List

import numpy as np

import ironic as ir
from geom import Quaternion, Transform3D
from tools.buttons import ButtonHandler

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("teleop.log", mode="w")])


@ir.ironic_system(input_ports=["teleop_transform", "teleop_buttons"],
                  input_props=["robot_position"],
                  output_ports=[
                      "robot_target_position",
                      "gripper_target_grasp",
                      "start_tracking",
                      "stop_tracking",
                      "start_recording",
                      "stop_recording",
                      "reset"
                  ],
                  output_props=["metadata"])
class TeleopSystem(ir.ControlSystem):
    def __init__(self, operator_position: str = 'back'):
        super().__init__()
        self.teleop_t = None
        self.offset = None
        self.operator_position = operator_position
        self.is_tracking = False
        self.is_recording = False
        self.button_handler = ButtonHandler()
        self.fps = ir.utils.FPSCounter("Teleop")

    def _parse_position(self, value: Transform3D) -> Transform3D:
        pos = np.array([value.translation[2], value.translation[0], value.translation[1]])
        quat = Quaternion(value.quaternion[0], value.quaternion[3], value.quaternion[1], value.quaternion[2])

        # Don't ask my why these transformations, I just got them
        # Rotate quat 90 degrees around Y axis
        res_quat = quat
        rotation_y_90 = Quaternion(np.cos(-np.pi/4), 0, np.sin(-np.pi/4), 0)
        res_quat = rotation_y_90 * quat

        if self.operator_position == 'back':
            pos[0] *= -1
            pos[1] *= -1
            res_quat = Quaternion(res_quat[0], -res_quat[1], res_quat[2], res_quat[3])
        elif self.operator_position == 'front':
            res_quat = Quaternion(-res_quat[0], res_quat[1], res_quat[2], res_quat[3])
        else:
            raise ValueError(f"Invalid operator position: {self.operator_position}")

        return Transform3D(pos, res_quat)


    def _parse_buttons(self, value: List[float]):
        if len(value) > 6:
            but = {
                'A': value[4],
                'B': value[5],
                'trigger': value[0],
                'thumb': value[1],
                'stick': value[3]
            }

            self.button_handler.update_buttons(but)

    @ir.out_property
    async def metadata(self):
        return {}

    @ir.on_message("teleop_transform")
    async def handle_teleop_transform(self, message: ir.Message):
        self.teleop_t = self._parse_position(message.data)
        self.fps.tick()

        if self.is_tracking and self.offset is not None:
            target = Transform3D(
                self.teleop_t.translation + self.offset.translation,
                self.teleop_t.quaternion * self.offset.quaternion
            )
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
            self.offset = Transform3D(
                -self.teleop_t.translation + robot_t.translation,
                self.teleop_t.quaternion.inv * robot_t.quaternion
            )
        if self.is_tracking:
            logging.info('Stopped tracking')
            self.is_tracking = False
            await self.outs.stop_tracking.write(ir.Message(None, timestamp))
        else:
            logging.info('Started tracking')
            self.is_tracking = True
            await self.outs.start_tracking.write(ir.Message(None, timestamp))

    async def _switch_recording(self, timestamp: int):
        if self.is_recording:
            logging.info('Stopped recording')
            self.is_recording = False
            await self.outs.stop_recording.write(ir.Message(None, timestamp))
        else:
            logging.info('Started recording')
            self.is_recording = True
            await self.outs.start_recording.write(ir.Message(None, timestamp))

