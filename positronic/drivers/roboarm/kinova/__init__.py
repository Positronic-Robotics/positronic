# In order to use Kinova API, you need to install the Kinova Kortex
# Go to https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.7.0/
# Download the wheel file and install it using pip install kortex_api-2.7.0-py3-none-any.whl

from collections import deque
from enum import Enum
import time
import asyncio
import multiprocessing as mp
from multiprocessing import shared_memory
import os

import numpy as np

import ironic as ir
from positronic.drivers.roboarm.kinova.api import KinovaAPI
from positronic.drivers.roboarm.kinova.base import KinematicsSolver, JointCompliantController
from positronic.drivers.roboarm.status import RobotStatus

_Q_RETRACT = np.array([0.0, -0, 0.5, -1.5, 0.0, -0.5, 1.57079633])


@ir.ironic_system(input_ports=['target_position', 'reset', 'target_grip'],
                  output_ports=['status'],
                  output_props=['position', 'joint_positions', 'grip', 'metadata'])
class Kinova(ir.ControlSystem):
    """Main control system interface for the Kinova arm."""

    class Priority(Enum):
        BLOCKING = 'blocking'
        NON_BLOCKING = 'non-blocking'

    class Status(Enum):
        RUNNING = 'running'
        COMPLETED = 'finished'
        ABORTED = 'aborted'

    def __init__(self, ip, relative_dynamics_factor=0.5):
        super().__init__()
        self._main_loop = None
        self.solver = KinematicsSolver()

        self.ip = ip
        self.control_process = None
        self.stop_event = mp.Event()
        self.joint_controller = None
        self.relative_dynamics_factor = relative_dynamics_factor

        self.command_queue = mp.Queue()
        self.shared_current = shared_memory.SharedMemory(create=True, size=7 * 4)
        self.command_finished = mp.Value('b', False)

        self.blocking_queue = deque()
        self.non_blocking_command = None
        self.running_command = None

    async def setup(self):
        self.control_process = mp.Process(target=self._control_loop, args=(self.command_queue, ))
        self.control_process.start()
        # Set roboarm in initial position
        await self.handle_reset(None)
        await self.step()

    async def cleanup(self):
        self.stop_event.set()
        if self.control_process and self.control_process.is_alive():
            self.control_process.join(timeout=5.0)
            if self.control_process.is_alive():
                self.control_process.terminate()

        self.shared_current.close()
        self.shared_current.unlink()
        self.shared_current = None

    @ir.on_message('reset')
    async def handle_reset(self, _message: ir.Message):
        """Commands the robot to move to the home position."""
        reset_future = asyncio.get_running_loop().create_future()
        self._submit_command(_Q_RETRACT, self.Priority.BLOCKING, reset_future)

        async def handle_status_transitions():
            await self.outs.status.write(ir.Message(RobotStatus.RESETTING, ir.system_clock()))
            await reset_future
            await self.outs.status.write(ir.Message(RobotStatus.AVAILABLE, ir.system_clock()))

        asyncio.get_running_loop().create_task(handle_status_transitions())

    @ir.on_message('target_position')
    async def handle_target_position(self, message: ir.Message):
        """Handles a target position message."""
        qpos = self.solver.inverse(message.data, self._q)
        self._submit_command(qpos, self.Priority.NON_BLOCKING, None)

    @ir.out_property
    async def position(self):
        """End effector position in robot base coordinate frame."""
        return ir.Message(self.solver.forward(self._q))

    @ir.on_message('target_grip')
    async def handle_target_grip(self, _message: ir.Message):
        pass

    @ir.out_property
    async def grip(self):
        return ir.Message(0.0)

    @ir.out_property
    async def metadata(self):
        return ir.Message({'env.arm': 'kinova'})

    @ir.out_property
    async def joint_positions(self):
        """Current joint positions."""
        return ir.Message(self._q)

    @property
    def _q(self):
        return np.ndarray((7, ), dtype=np.float32, buffer=self.shared_current.buf).copy()

    async def step(self):
        finished = False
        with self.command_finished.get_lock():
            if self.command_finished.value and self.running_command is not None:
                finished = True
                self.command_finished.value = False

        if finished:
            future = self.running_command[2]
            if future is not None:
                future.set_result(Kinova.Status.COMPLETED)
            self.running_command = None

        has_new_commands = len(self.blocking_queue) > 0 or self.non_blocking_command is not None
        if self.running_command is not None:
            _, priority, running_future = self.running_command
            if priority == self.Priority.NON_BLOCKING and has_new_commands:
                if running_future is not None:
                    running_future.set_result(Kinova.Status.ABORTED)
                self.running_command = None

        if self.running_command is None and has_new_commands:
            if self.blocking_queue:
                qpos, future = self.blocking_queue.popleft()
                self.running_command = qpos, self.Priority.BLOCKING, future
                self.command_queue.put(qpos)
            elif self.non_blocking_command is not None:
                qpos, future = self.non_blocking_command
                self.non_blocking_command = None
                self.running_command = qpos, self.Priority.NON_BLOCKING, future
                self.command_queue.put(qpos)

        return ir.State.ALIVE

    def _submit_command(self, qpos, priority, future):
        self.non_blocking_command = None
        if priority == self.Priority.BLOCKING:
            self.blocking_queue.append((qpos, future))
        else:
            self.non_blocking_command = (qpos, future)

    def _control_loop(self, command_queue):
        try:
            # Set realtime scheduling priority
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
            print("Successfully set realtime scheduling priority")
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not set realtime scheduling priority: {e}")
            print("Run `sudo setcap 'cap_sys_nice=eip' $(which python3)` to enable this")
            print("Control loop will run with normal scheduling")

        # Note: Torque commands are converted to current commands since
        # Kinova's torque controller is unable to achieve commanded torques.
        # See relevant GitHub issue: https://github.com/Kinovarobotics/kortex/issues/38
        torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
        current_array = np.ndarray((7, ), dtype=np.float32, buffer=self.shared_current.buf)

        with KinovaAPI(self.ip) as api:
            fps = ir.utils.FPSCounter('Kinova', report_every_sec=3.0)

            joint_controller = JointCompliantController(api.actuator_count, self.relative_dynamics_factor)
            current_command = np.zeros(api.actuator_count, dtype=np.float32)

            last_ts = time.monotonic()
            q, dq, tau = api.apply_current_command(None)  # Warm up
            joint_controller.compute_torque(q, dq, tau)

            while not self.stop_event.is_set() or not joint_controller.finished:
                now_ts = time.monotonic()
                step_time = now_ts - last_ts
                if step_time > 0.005:  # 5 ms
                    print(f'Warning: Step time {1000 * step_time:.3f} ms')
                last_ts = now_ts

                # Don't consume target if we're stopping, so as soon as we reach the target the loop stops
                while not command_queue.empty() and not self.stop_event.is_set():
                    qpos = command_queue.get()
                    joint_controller.set_target_qpos(qpos)

                torque_command = joint_controller.compute_torque(q, dq, tau)
                current_array[:] = joint_controller.q_s
                np.divide(torque_command, torque_constant, out=current_command)
                q, dq, tau = api.apply_current_command(current_command)

                if joint_controller.finished:
                    with self.command_finished.get_lock():
                        self.command_finished.value = True

                fps.tick()


class KinovaSync:
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.5):
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self._control_process = None
        self.command_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.command_finished = mp.Value('b', True)
        self.shared_current = shared_memory.SharedMemory(create=True, size=7 * 4)
        self.solver = KinematicsSolver()

    def _control_loop(self, command_queue):
        try:
            # Set realtime scheduling priority
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
            print("Successfully set realtime scheduling priority")
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not set realtime scheduling priority: {e}")
            print("Run `sudo setcap 'cap_sys_nice=eip' $(which python3)` to enable this")
            print("Control loop will run with normal scheduling")

        # Note: Torque commands are converted to current commands since
        # Kinova's torque controller is unable to achieve commanded torques.
        # See relevant GitHub issue: https://github.com/Kinovarobotics/kortex/issues/38
        torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
        current_array = np.ndarray((7, ), dtype=np.float32, buffer=self.shared_current.buf)

        with KinovaAPI(self.ip) as api:
            fps = ir.utils.FPSCounter('Kinova', report_every_sec=3.0)

            joint_controller = JointCompliantController(api.actuator_count, self.relative_dynamics_factor)
            current_command = np.zeros(api.actuator_count, dtype=np.float32)
            qpos = np.zeros(7, dtype=np.float32)
            last_ts = time.monotonic()
            q, dq, tau = api.apply_current_command(None)  # Warm up
            joint_controller.compute_torque(q, dq, tau)
            while not self.stop_event.is_set() or not joint_controller.finished:
                now_ts = time.monotonic()
                step_time = now_ts - last_ts
                if step_time > 0.005:  # 5 ms
                    print(f'Warning: Step time {1000 * step_time:.3f} ms')
                last_ts = now_ts

                # Don't consume target if we're stopping, so as soon as we reach the target the loop stops
                while not command_queue.empty() and not self.stop_event.is_set():
                    qpos = command_queue.get()
                    joint_controller.set_target_qpos(qpos)

                torque_command = joint_controller.compute_torque(q, dq, tau)
                current_array[:] = joint_controller.q_s
                np.divide(torque_command, torque_constant, out=current_command)
                q, dq, tau = api.apply_current_command(current_command)

                if joint_controller.finished:
                    with self.command_finished.get_lock():
                        self.command_finished.value = True

                fps.tick()

    def setup(self):
        self._control_process = mp.Process(target=self._control_loop, args=(self.command_queue, ))
        self._control_process.start()

    def cleanup(self):
        self.stop_event.set()
        if self._control_process and self._control_process.is_alive():
            self._control_process.join(timeout=5.0)

    def execute_joint_command(self, qpos: np.ndarray):
        self.wait_finish()

        with self.command_finished.get_lock():
            self.command_finished.value = False

        self.command_queue.put(qpos)

    def wait_finish(self):
        time.sleep(0.2)  # TODO: figure out why adding it here improves policy accuracy
        while True:
            with self.command_finished.get_lock():
                if self.command_finished.value:
                    break
            time.sleep(0.001)

    def execute_cartesian_command(self, position: np.ndarray):
        self.execute_joint_command(self.solver.inverse(position, self._q))

    def reset_position(self):
        print('resetting')
        self.execute_joint_command(_Q_RETRACT)

    def get_position(self):
        return self.solver.forward(self._q)

    def get_joint_positions(self):
        return self._q

    @property
    def _q(self):
        return np.ndarray((7, ), dtype=np.float32, buffer=self.shared_current.buf).copy()
