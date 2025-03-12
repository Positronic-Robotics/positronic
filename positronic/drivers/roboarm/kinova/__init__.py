# This file contains code derived from tidybot2, which is:
# Copyright (c) 2024 Jimmy Wu
# Released under MIT License (https://github.com/jimmyyhwu/tidybot2/blob/main/LICENSE)
#
# All modifications and additions are:
# Copyright (c) 2025 Positronic Robotics Inc.
# All rights reserved.
# This code may not be used, copied, modified, merged, published, distributed,
# sublicensed, and/or sold without explicit permission from the copyright holder.

# In order to use Kinova API, you need to install the Kinova Kortex
# Go to https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.7.0/
# Download the wheel file and install it using pip install kortex_api-2.7.0-py3-none-any.whl

from collections import deque
from enum import Enum
import math
import threading
import time
import asyncio
import multiprocessing as mp
from multiprocessing import shared_memory
import os

import mujoco
import numpy as np
import pinocchio as pin
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import (ActuatorCyclic_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2,
                                         Session_pb2)
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

import geom
import ironic as ir
from positronic.drivers.roboarm.status import RobotStatus

_TCP_PORT = 10000
_UDP_PORT = 10001

K_r = np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18])
K_l = np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0])
K_lp = np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
K_r_inv = np.linalg.inv(K_r)
K_r_K_l = K_r @ K_l
_DT = 0.001

_DAMPING_COEFF = 1e-12
_MAX_ANGLE_CHANGE = np.deg2rad(45)

_Q_RETRACT = np.array([0.0, -0.34906585, 3.14159265, -2, 0.0, -1.42, 1.57079633])


def _wrap_joint_angle(q, q_base):
    return q_base + np.mod(q - q_base + np.pi, 2 * np.pi) - np.pi


class DeviceConnection:
    """Manages TCP/UDP connections to a Kinova device."""

    def __init__(self,
                 ip_address: str,
                 port: int,
                 transport: TCPTransport | UDPTransport,
                 credentials: tuple[str, str] = ('admin', 'admin')):
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = transport
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != '':
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager = SessionManager(self.router)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()


class KinematicsSolver:
    """Solves forward and inverse kinematics for the Kinova arm."""

    def __init__(self, ee_offset=0.0):
        self.model = mujoco.MjModel.from_xml_path('positronic/drivers/roboarm/kinova/gen3.xml')
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        self.qpos0 = self.model.key('retract').qpos  # TODO: Is it good for IK null space?
        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Add end effector offset for gripper
        # 0.061525 comes from the Kinova URDF
        self.model.site(self.site_id).pos = np.array([0.0, 0.0, -0.061525 - ee_offset])

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = _DAMPING_COEFF * np.eye(6)
        self.eye = np.eye(self.model.nv)

    def forward(self, qpos):
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        pos = self.data.site(self.site_id).xpos.copy()
        mat = self.data.site(self.site_id).xmat.copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat)
        return geom.Transform3D(pos, geom.Rotation.from_quat(quat))

    def inverse(self, pos: geom.Transform3D, qpos0: np.ndarray, max_iters: int = 20, err_thresh: float = 1e-4):
        self.data.qpos = qpos0

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos.translation - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, pos.rotation.as_quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            if np.linalg.norm(self.err) < err_thresh:
                break

            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (self.eye -
                       (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > _MAX_ANGLE_CHANGE:
                update *= _MAX_ANGLE_CHANGE / update_max

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()


class JointCompliantController:
    """Implements compliant joint control for the Kinova arm."""

    class LowPassFilter:
        """Simple low-pass filter implementation."""

        def __init__(self, alpha, initial_value):
            assert 0 < alpha <= 1, 'Alpha must be between 0 and 1'
            self.alpha = alpha
            self.y = initial_value

        def filter(self, x):
            self.y = self.alpha * x + (1 - self.alpha) * self.y
            return self.y

    def __init__(self, actuator_count, relative_dynamics_factor=0.5):
        self.q_s = None
        self.q_d = None
        self.dq_d = None
        self.q_n = None
        self.dq_n = None
        self.tau_filter = None

        self.actuator_count = actuator_count
        self.otg = None
        self.otg_inp = None
        self.otg_out = None
        self.otg_res = None
        self.relative_dynamics_factor = relative_dynamics_factor

        self.target_qpos = None

        # Initialize pinocchio model and data
        self.model = pin.buildModelFromUrdf('positronic/drivers/roboarm/kinova/model.urdf')
        self.data = self.model.createData()
        self._q_pin = np.zeros(11)

    def set_target_qpos(self, qpos):
        self.target_qpos = qpos

    @property
    def finished(self):
        return self.otg_res == Result.Finished

    def compute_torque(self, q, dq, tau):
        q_pin = self._q_pin  # Reuse pre-allocated q_pin
        q_pin[0], q_pin[1], q_pin[2] = math.cos(q[0]), math.sin(q[0]), q[1]
        q_pin[3], q_pin[4], q_pin[5] = math.cos(q[2]), math.sin(q[2]), q[3]
        q_pin[6], q_pin[7], q_pin[8] = math.cos(q[4]), math.sin(q[4]), q[5]
        q_pin[9], q_pin[10] = math.cos(q[6]), math.sin(q[6])

        gravity = pin.computeGeneralizedGravity(self.model, self.data, q_pin)

        # Initialize controller state if needed
        if self.q_s is None:
            self.q_s = q.copy()
            self.q_d = q.copy()
            self.dq_d = np.zeros_like(q)
            self.q_n = q.copy()
            self.dq_n = dq.copy()
            self.tau_filter = JointCompliantController.LowPassFilter(0.01, tau.copy())

            self.otg = Ruckig(self.actuator_count, _DT)
            self.otg_inp = InputParameter(self.actuator_count)
            self.otg_out = OutputParameter(self.actuator_count)
            self.otg_inp.max_velocity = (4 * [math.radians(80 * self.relative_dynamics_factor)] +
                                         3 * [math.radians(140 * self.relative_dynamics_factor)])
            self.otg_inp.max_acceleration = (4 * [math.radians(240 * self.relative_dynamics_factor)] +
                                             3 * [math.radians(450 * self.relative_dynamics_factor)])
            self.otg_inp.current_position = q.copy()
            self.otg_inp.current_velocity = dq.copy()
            self.otg_inp.target_position = q.copy()
            self.otg_inp.target_velocity = np.zeros(self.actuator_count)
            self.otg_res = Result.Finished

        self.q_s = _wrap_joint_angle(q, self.q_s)
        dq_s = dq.copy()  # TODO: It seems that we don't need copy here
        tau_s_f = self.tau_filter.filter(tau)

        if self.target_qpos is not None:
            qpos = _wrap_joint_angle(self.target_qpos, self.q_s)
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

            self.target_qpos = None

        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.q_d[:] = self.otg_out.new_position
            self.dq_d[:] = self.otg_out.new_velocity

        tau_task = -K_p @ (self.q_n - self.q_d) - K_d @ (self.dq_n - self.dq_d) + gravity

        # Nominal motor plant
        ddq_n = K_r_inv @ (tau_task - tau_s_f)
        self.dq_n += ddq_n * _DT
        self.q_n += self.dq_n * _DT

        tau_f = K_r_K_l @ ((self.dq_n - dq_s) + K_lp @ (self.q_n - self.q_s))  # Nominal friction

        return tau_task + tau_f


class KinovaAPI:
    """Low-level interface to the Kinova arm hardware."""

    def __init__(self, ip):
        self.ip = ip
        self.tcp_connection = DeviceConnection(ip, _TCP_PORT, TCPTransport())
        self.udp_connection = DeviceConnection(ip, _UDP_PORT, UDPTransport())

        self.base = None
        self.base_cyclic = None
        self.actuator_config = None
        self.actuator_count = None
        self.actuator_device_ids = None

        self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
        self.current_limit_min = -self.current_limit_max

        self.base_feedback, self.base_command = None, None

    def __enter__(self):
        tcp_router = self.tcp_connection.__enter__()
        udp_router = self.udp_connection.__enter__()
        self.base = BaseClient(tcp_router)
        self.base_cyclic = BaseCyclicClient(udp_router)
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count

        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]

        # Configure communication
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 30

        # Make sure actuators are in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        if self.base.GetArmState().active_state == Base_pb2.ARMSTATE_IN_FAULT:
            self.base.ClearFaults()
            while self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY:
                time.sleep(0.1)

        # Initialize control loop
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))
        # Set actuators to current control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('CURRENT')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        self.base_feedback = self.base_cyclic.RefreshFeedback()
        self.base_command = BaseCyclic_pb2.Command()
        for i in range(self.actuator_count):
            self.base_command.actuators.add(flags=ActuatorCyclic_pb2.SERVO_ENABLE)

        return self

    def __exit__(self, *_):
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Set arm back to high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()

    def apply_current_command(self, current_command):
        if current_command is not None:
            np.clip(current_command, self.current_limit_min, self.current_limit_max, out=current_command)

        # Increment frame ID to ensure actuators can reject out-of-order frames
        self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536
        for i in range(self.actuator_count):
            self.base_command.actuators[i].current_motor = (current_command[i] if current_command is not None else
                                                            self.base_feedback.actuators[i].current_motor)
            self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
            self.base_command.actuators[i].command_id = self.base_command.frame_id

        self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.send_options)

        q, dq, tau = np.zeros(self.actuator_count), np.zeros(self.actuator_count), np.zeros(self.actuator_count)

        for i in range(self.actuator_count):
            q[i] = self.base_feedback.actuators[i].position
            dq[i] = self.base_feedback.actuators[i].velocity
            tau[i] = self.base_feedback.actuators[i].torque

        np.deg2rad(q, out=q)
        np.deg2rad(dq, out=dq)
        np.negative(tau, out=tau)  # Raw torque readings are negative relative to actuator direction
        return q, dq, tau


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
        FINISHED = 'finished'
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
                future.set_result(Kinova.Status.FINISHED)  # TODO: Better status
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
