import logging
from typing import List
import threading

import numpy as np
import pyspacemouse

import ironic as ir
from geom import Rotation, Transform3D


@ir.ironic_system(
    input_props=["robot_position"],
    output_ports=["robot_target_position", "gripper_target_grasp", "start_recording", "stop_recording", "reset"])
class SpacemouseCS(ir.ControlSystem):

    def __init__(
        self,
        translation_speed: float = 0.0005,
        rotation_speed: float = 0.001,
        translation_dead_zone: float = 0.8,
        rotation_dead_zone: float = 0.7,
    ):
        super().__init__()
        self.translation_speed = translation_speed
        self.translation_dead_zone = translation_dead_zone
        self.rotation_speed = rotation_speed
        self.rotation_dead_zone = rotation_dead_zone

        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_data = None

        self.teleop_delta = Transform3D()
        self.initial_position = None
        self.is_tracking = False

        self.buttons = [False, False]

        self.fps = ir.utils.FPSCounter("Spacemouse")

    async def setup(self):
        pyspacemouse.open()
        self.thread.start()

    async def step(self):
        with self.state_lock:
            state = self.latest_data
        pressed = self._get_pressed_buttons(state)
        self.buttons = [bool(state.buttons[0]), bool(state.buttons[1])]

        if pressed[0]:
            await self._switch_tracking()

        if self.is_tracking:
            if self.initial_position is None:
                robot_t = (await self.ins.robot_position()).data
                self.initial_position = Transform3D(robot_t.translation, robot_t.quaternion)

            xyz = np.array([state.x, state.y, state.z])
            xyz[np.abs(xyz) < self.translation_dead_zone] = 0

            rpy = np.array([-state.yaw, state.roll, state.pitch])
            rpy[np.abs(rpy) < self.rotation_dead_zone] = 0

            self.teleop_delta = Transform3D(
                self.teleop_delta.translation + xyz * self.translation_speed,
                self.teleop_delta.rotation * Rotation.from_euler(rpy * self.rotation_speed))

            new_position = Transform3D(self.teleop_delta.translation + self.initial_position.translation,
                                       self.initial_position.rotation * self.teleop_delta.rotation)

            await self.outs.gripper_target_grasp.write(ir.Message(1.0 - state.buttons[1], ir.system_clock()))
            await self.outs.robot_target_position.write(ir.Message(new_position, ir.system_clock()))

        return ir.State.ALIVE

    def _get_pressed_buttons(self, state: pyspacemouse.SpaceNavigator) -> List[bool]:
        return [not self.buttons[0] and bool(state.buttons[0]), not self.buttons[1] and bool(state.buttons[1])]

    async def _switch_tracking(self):
        # TODO: This resembles the teleop system, maybe we should make a generic system for this
        if self.is_tracking:
            logging.info('Stopped tracking')
            self.is_tracking = False
        else:
            logging.info('Started tracking')
            self.is_tracking = True

    def _read_spacemouse(self):
        while not self.stop_event.is_set():
            state = pyspacemouse.read()
            self.fps.tick()
            with self.state_lock:
                self.latest_data = state
        pyspacemouse.close()

    async def cleanup(self):
        self.stop_event.set()
        self.thread.join()
