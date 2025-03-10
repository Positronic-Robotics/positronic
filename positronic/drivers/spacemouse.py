import logging
import time
from typing import List
import multiprocessing as mp
from multiprocessing import shared_memory

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

        # Create shared memory for spacemouse state
        # Format: [x, y, z, roll, pitch, yaw, button1, button2]
        self.shm = shared_memory.SharedMemory(create=True, size=8 * 8)  # 8 floats, 8 bytes each
        self.shared_array = np.ndarray((8,), dtype=np.float64, buffer=self.shm.buf)
        self.shared_array[:] = 0  # Initialize with zeros

        # Create a process instead of a thread
        self.stop_event = mp.Event()
        self.process = mp.Process(target=self._read_spacemouse, args=(self.shm.name, self.stop_event), daemon=True)

        self.teleop_delta = Transform3D()
        self.initial_position = None
        self.is_tracking = False

        self.buttons = [False, False]

        self.fps = ir.utils.FPSCounter("Spacemouse")

    async def setup(self):
        self.process.start()

    async def step(self):
        # Read from shared memory
        state_array = np.copy(self.shared_array)

        # Convert shared memory data to a state-like object
        class StateProxy:
            def __init__(self, data):
                self.x = data[0]
                self.y = data[1]
                self.z = data[2]
                self.roll = data[3]
                self.pitch = data[4]
                self.yaw = data[5]
                self.buttons = [bool(data[6]), bool(data[7])]

        state = StateProxy(state_array)

        pressed = self._get_pressed_buttons(state)
        self.buttons = [bool(state.buttons[0]), bool(state.buttons[1])]

        if pressed[0]:
            await self._switch_tracking()

        if pressed[1]:
            await self.outs.reset.write(ir.Message(True))

        if self.is_tracking:
            if self.initial_position is None:
                robot_t = (await self.ins.robot_position()).data
                self.initial_position = robot_t.copy()

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

    @staticmethod
    def _read_spacemouse(shm_name, stop_event):
        # Open the spacemouse in this process
        pyspacemouse.open()

        # Connect to the shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray((8,), dtype=np.float64, buffer=shm.buf)

        fps_counter = ir.utils.FPSCounter("Spacemouse")

        try:
            while not stop_event.is_set():
                state = pyspacemouse.read()
                fps_counter.tick()

                if state is not None:
                    # Update shared memory with new state
                    shared_array[0] = state.x
                    shared_array[1] = state.y
                    shared_array[2] = state.z
                    shared_array[3] = state.roll
                    shared_array[4] = state.pitch
                    shared_array[5] = state.yaw
                    shared_array[6] = float(state.buttons[0])
                    shared_array[7] = float(state.buttons[1])

                time.sleep(0.01)
        finally:
            pyspacemouse.close()
            shm.close()

    async def cleanup(self):
        if self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()

        # Clean up shared memory
        self.shm.close()
        self.shm.unlink()
