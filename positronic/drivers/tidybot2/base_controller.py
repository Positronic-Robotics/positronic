# MIT License

# Copyright (c) 2024 Jimmy Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Jimmy Wu
# Date: October 2024
#
# IMPORTANT: The mobile base has very strong motors and should be closely
# monitored during operation, especially if any changes are made to this code.
# When testing out new code, consider starting out with very low speeds and
# turning the mobile base on its side to elevate the casters off the ground.
# Torque current limits are used to limit the output torque of the motors.
#
# If you will be developing on this controller rather than just running it,
# additional safety measures are highly recommended:
# - Always use an enabling device when operating the robot
# - Install safety barriers around the robot operating area
# - Install a physical e-stop button on the robot
# - Install bump sensors on the robot to detect collisions
#
# References:
# - https://github.com/google/powered-caster-vehicle

# Tell Phoenix 6 to use hardware instead of simulation
import os

os.environ['CTR_TARGET'] = 'Hardware'  # pylint: disable=wrong-import-position

import math
import multiprocessing as mp
import time
from enum import Enum

import numpy as np
import phoenix6
from phoenix6 import configs, controls, hardware
from ruckig import (ControlInterface, InputParameter, OutputParameter, Result,
                    Ruckig)
from threadpoolctl import threadpool_limits

from positronic.drivers.tidybot2.constants import POLICY_CONTROL_PERIOD, h_x, h_y
from positronic.drivers.tidybot2.utils import create_pid_file

# Vehicle
CONTROL_FREQ = 250  # 250 Hz
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms
NUM_CASTERS = 4

# Caster
b_x = -0.014008  # Caster offset (m)
b_y = -0.003753  # Lateral caster offset (m)
r = 0.0508  # Wheel radius (m)
N_s = 32.0 / 15.0 * 60.0 / 10.0  # Steer gear ratio
N_r1 = 50.0 / 14.0  # Drive gear ratio (1st stage)
N_r2 = 19.0 / 25.0  # Drive gear ratio (2nd stage)
N_w = 45.0 / 15.0  # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w
N_s_r2_w = N_s * N_r2 * N_w
TWO_PI = 2 * math.pi


class Motor:

    def __init__(self, num):
        self.num = num
        self.is_steer = num % 2 != 0  # Odd num motors are steer motors
        self.fx = hardware.TalonFX(self.num)
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC
        self.fx.set_position(0)
        fx_cfg = configs.TalonFXConfiguration()

        if self.num == 1:
            time.sleep(0.2)  # First CAN device, wait for CAN bus to be ready
            supply_voltage = self.fx.get_supply_voltage().value
            print(f'Motor supply voltage: {supply_voltage:.2f} V')
            if supply_voltage < 11.5:
                raise Exception('Motor supply voltage is too low. Please charge the battery.')

        # Status signals
        self.position_signal = self.fx.get_position()
        self.velocity_signal = self.fx.get_velocity()
        self.status_signals = [self.position_signal, self.velocity_signal]

        # Control requests
        self.velocity_request = controls.VelocityTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Velocity control gains
        fx_cfg.slot0.k_p = 5.0
        fx_cfg.slot0.k_d = 0.1 if self.is_steer else 0.0  # Set k_d for steer to prevent caster flutter

        # Current limits (hard floor with no incline)
        # IMPORTANT: These values limit the force that the base can generate. Proceed very carefully if modifying these values.
        torque_current_limit = 40 if self.is_steer else 10  # 40 A for steer, 10 A for drive
        fx_cfg.torque_current.peak_forward_torque_current = torque_current_limit
        fx_cfg.torque_current.peak_reverse_torque_current = -torque_current_limit

        # Disable beeps (Note: beep_on_config is not yet supported as of Oct 2024)
        fx_cfg.audio.beep_on_boot = False
        # fx_cfg.audio.beep_on_config = False

        # Apply configuration
        self.fx.configurator.apply(fx_cfg)

    def get_position(self):
        return TWO_PI * self.position_signal.value

    def get_velocity(self):
        return TWO_PI * self.velocity_signal.value

    def set_velocity(self, velocity):
        self.fx.set_control(self.velocity_request.with_velocity(velocity / TWO_PI))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)


class Caster:

    def __init__(self, num, magnet_offset):
        self.num = num
        self.steer_motor = Motor(2 * self.num - 1)
        self.drive_motor = Motor(2 * self.num)
        self.cancoder = hardware.CANcoder(self.num)
        self.cancoder_cfg = configs.CANcoderConfiguration()

        # Status signals
        self.steer_position_signal = self.cancoder.get_absolute_position()
        self.steer_velocity_signal = self.cancoder.get_velocity()
        self.status_signals = self.steer_motor.status_signals + self.drive_motor.status_signals
        self.status_signals.extend([self.steer_position_signal, self.steer_velocity_signal])

        # Encoder magnet offset
        self.cancoder_cfg.magnet_sensor.magnet_offset = magnet_offset
        self.cancoder.configurator.apply(self.cancoder_cfg)

    def get_steer_position(self):
        return TWO_PI * self.steer_position_signal.value

    def get_steer_velocity(self):
        return TWO_PI * self.steer_velocity_signal.value

    def get_positions(self):
        steer_motor_pos = self.steer_motor.get_position()
        drive_motor_pos = self.drive_motor.get_position()
        # steer_pos = steer_motor_pos / N_s
        steer_pos = self.get_steer_position()
        drive_pos = steer_motor_pos / N_s_r2_w - drive_motor_pos / N_r1_r2_w
        return steer_pos, drive_pos

    def get_velocities(self):
        steer_motor_vel = self.steer_motor.get_velocity()
        drive_motor_vel = self.drive_motor.get_velocity()
        steer_vel = steer_motor_vel / N_s
        # steer_vel = self.get_steer_velocity()  # Very noisy
        drive_vel = steer_motor_vel / N_s_r2_w - drive_motor_vel / N_r1_r2_w
        return steer_vel, drive_vel

    def set_velocities(self, steer_vel, drive_vel):
        self.steer_motor.set_velocity(N_s * steer_vel)
        self.drive_motor.set_velocity(N_r1 * steer_vel - N_r1_r2_w * drive_vel)

    def set_neutral(self):
        self.steer_motor.set_neutral()
        self.drive_motor.set_neutral()


class CommandType(Enum):
    POSITION = 'position'
    VELOCITY = 'velocity'


# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'


class Vehicle:
    """Vehicle class that can be used to control the mobile base."""

    def __init__(self, encoder_magnet_offsets, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        max_vel = np.array(max_vel)
        max_accel = np.array(max_accel)

        self.command_queue = mp.Queue(1)
        self.control_loop_process = mp.Process(target=self.control_loop,
                                               args=(max_vel, max_accel, encoder_magnet_offsets))
        self.control_loop_running = mp.Event()

    def start_control(self):
        if self.control_loop_process is None:
            print('To initiate a new control loop, please create a new instance of Vehicle.')
            return
        self.control_loop_running.set()
        self.control_loop_process.start()

    def stop_control(self):
        self.control_loop_running.clear()
        self.control_loop_process.join(timeout=5)
        if self.control_loop_process.is_alive():
            self.control_loop_process.terminate()
            self.control_loop_process.join()
        self.control_loop_process = None

    def control_loop(self, max_vel, max_accel, encoder_magnet_offsets):
        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        except PermissionError:
            print('Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf')
            raise

        vehicle = InternalVehicle(max_vel, max_accel, encoder_magnet_offsets)
        print('>>>>>>> Control loop started <<<<<<')

        disable_motors = True
        last_command_time = time.time()
        last_step_time = time.time()
        while self.control_loop_running.is_set():
            # Maintain the desired control frequency
            while time.time() - last_step_time < CONTROL_PERIOD:
                time.sleep(0.0001)
            curr_time = time.time()
            step_time = curr_time - last_step_time
            last_step_time = curr_time
            if step_time > 0.005:  # 5 ms
                print(f'Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop')

            # Update state
            vehicle.update_state()

            # Global to local frame conversion
            theta = vehicle.x[2]
            R = np.array([[math.cos(theta), math.sin(theta), 0.0],
                          [-math.sin(theta), math.cos(theta), 0.0],
                          [0.0, 0.0, 1.0]])

            # Check for new command
            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()
                target = command['target']

                # Velocity command
                if command['type'] == CommandType.VELOCITY:
                    if command['frame'] == FrameType.LOCAL:
                        target = R.T @ target
                    vehicle.otg_inp.control_interface = ControlInterface.Velocity
                    vehicle.otg_inp.target_velocity = np.clip(target, -max_vel, max_vel)

                # Position command
                elif command['type'] == CommandType.POSITION:
                    vehicle.otg_inp.control_interface = ControlInterface.Position
                    vehicle.otg_inp.target_position = target
                    vehicle.otg_inp.target_velocity = np.zeros_like(vehicle.dx)

                vehicle.otg_res = Result.Working
                disable_motors = False

            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                vehicle.otg_inp.target_position = vehicle.otg_out.new_position
                vehicle.otg_inp.target_velocity = np.zeros_like(vehicle.dx)
                vehicle.otg_inp.current_velocity = vehicle.dx  # Set this to prevent lurch when command stream resumes
                vehicle.otg_res = Result.Working
                disable_motors = True

            # Slow down base during caster flip
            # Note: At low speeds, this section can be disabled for smoother movement
            if np.max(np.abs(vehicle.dq[::2])) > 12.56:  # Steer joint speed > 720 deg/s
                if vehicle.otg_inp.control_interface == ControlInterface.Position:
                    vehicle.otg_inp.target_position = vehicle.otg_out.new_position
                elif vehicle.otg_inp.control_interface == ControlInterface.Velocity:
                    vehicle.otg_inp.target_velocity = np.zeros_like(vehicle.dx)

            # Update OTG
            if vehicle.otg_res == Result.Working:
                vehicle.otg_inp.current_position = vehicle.x
                vehicle.otg_res = vehicle.otg.update(vehicle.otg_inp, vehicle.otg_out)
                vehicle.otg_out.pass_to_input(vehicle.otg_inp)

            if disable_motors:
                # Send motor neutral commands
                for i in range(NUM_CASTERS):
                    vehicle.casters[i].set_neutral()

            else:
                # Send enable signal to devices
                phoenix6.unmanaged.feed_enable(0.1)

                # Operational space velocity
                dx_d = vehicle.otg_out.new_velocity
                dx_d_local = R @ dx_d

                # Joint velocities
                dq_d = vehicle.C @ dx_d_local

                # Send motor velocity commands
                for i in range(NUM_CASTERS):
                    vehicle.casters[i].set_velocities(dq_d[2 * i], dq_d[2 * i + 1])

            # Debugging
            # vehicle.data.append({
            #     'timestamp': time.time(),
            #     'q': vehicle.q.tolist(),
            #     'dq': vehicle.dq.tolist(),
            #     'x': vehicle.x.tolist(),
            #     'dx': vehicle.dx.tolist(),
            # })
            # vehicle.redis_client.set(f'x', f'{vehicle.x[0]} {vehicle.x[1]} {vehicle.x[2]}')
            # vehicle.redis_client.set(f'dx', f'{vehicle.dx[0]} {vehicle.dx[1]} {vehicle.dx[2]}')

    def _enqueue_command(self, command_type, target, frame=None):
        if not self.command_queue.full():
            command = {'type': command_type, 'target': target}
            if frame is not None:
                command['frame'] = FrameType(frame)
            self.command_queue.put(command, block=False)

    def set_target_velocity(self, velocity, frame='global'):
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position):
        self._enqueue_command(CommandType.POSITION, position)


class InternalVehicle:
    """Stripped down version of Vehicle from Tidybot2, that can be used inside control loop."""

    def __init__(self, max_vel, max_accel, encoder_magnet_offsets):
        self.max_vel = np.array(max_vel)
        self.max_accel = np.array(max_accel)

        # Use PID file to enforce single instance
        create_pid_file('tidybot2-base-controller')

        # Initialize casters
        self.casters = [Caster(num + 1, encoder_magnet_offsets[num]) for num in range(NUM_CASTERS)]

        # CAN bus update frequency
        self.status_signals = [signal for caster in self.casters for signal in caster.status_signals]
        phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        # Joint space
        num_motors = 2 * NUM_CASTERS
        self.q = np.zeros(num_motors)
        self.dq = np.zeros(num_motors)
        self.tau = np.zeros(num_motors)

        # Operational space (global frame)
        num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(num_dofs)
        self.dx = np.zeros(num_dofs)

        # C matrix relating operational space velocities to joint velocities
        self.C = np.zeros((num_motors, num_dofs))
        self.C_steer = self.C[::2]
        self.C_drive = self.C[1::2]

        # C_p matrix relating operational space velocities to wheel velocities at the contact points
        self.C_p = np.zeros((num_motors, num_dofs))
        self.C_p_steer = self.C_p[::2]
        self.C_p_drive = self.C_p[1::2]
        self.C_p_steer[:, :2] = [1.0, 0.0]
        self.C_p_drive[:, :2] = [0.0, 1.0]

        # C_qp^# matrix relating joint velocities to operational space velocities
        self.C_pinv = np.zeros((num_motors, num_dofs))
        self.CpT_Cqinv = np.zeros((num_dofs, num_motors))
        self.CpT_Cqinv_steer = self.CpT_Cqinv[:, ::2]
        self.CpT_Cqinv_drive = self.CpT_Cqinv[:, 1::2]

        # OTG (online trajectory generation)
        # Note: It would be better to couple x and y using polar coordinates
        self.otg = Ruckig(num_dofs, CONTROL_PERIOD)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_res = Result.Working
        self.otg_inp.max_velocity = self.max_vel
        self.otg_inp.max_acceleration = self.max_accel

    def update_state(self):
        # Update all status signals (sensor values)
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)  # Note: Signal latency is roughly 4 ms

        # Joint positions and velocities
        for i, caster in enumerate(self.casters):
            self.q[2 * i:2 * i + 2] = caster.get_positions()
            self.dq[2 * i:2 * i + 2] = caster.get_velocities()

        q_steer = self.q[::2]
        s = np.sin(q_steer)
        c = np.cos(q_steer)

        # C matrix
        self.C_steer[:, 0] = s / b_x
        self.C_steer[:, 1] = -c / b_x
        self.C_steer[:, 2] = (-h_x * c - h_y * s) / b_x - 1.0
        self.C_drive[:, 0] = c / r - b_y * s / (b_x * r)
        self.C_drive[:, 1] = s / r + b_y * c / (b_x * r)
        self.C_drive[:, 2] = (h_x * s - h_y * c) / r + b_y * (h_x * c + h_y * s) / (b_x * r)

        # C_p matrix
        self.C_p_steer[:, 2] = -b_x * s - b_y * c - h_y
        self.C_p_drive[:, 2] = b_x * c - b_y * s + h_x

        # C_qp^# matrix
        self.CpT_Cqinv_steer[0] = b_x * s + b_y * c
        self.CpT_Cqinv_steer[1] = -b_x * c + b_y * s
        self.CpT_Cqinv_steer[2] = b_x * (-h_x * c - h_y * s - b_x) + b_y * (h_x * s - h_y * c - b_y)
        self.CpT_Cqinv_drive[0] = r * c
        self.CpT_Cqinv_drive[1] = r * s
        self.CpT_Cqinv_drive[2] = r * (h_x * s - h_y * c - b_y)
        with threadpool_limits(limits=1, user_api='blas'):  # Prevent excessive CPU usage
            self.C_pinv = np.linalg.solve(self.C_p.T @ self.C_p, self.CpT_Cqinv)

        # Odometry
        dx_local = self.C_pinv @ self.dq
        theta_avg = self.x[2] + 0.5 * dx_local[2] * CONTROL_PERIOD
        R = np.array([[math.cos(theta_avg), -math.sin(theta_avg), 0.0], [math.sin(theta_avg),
                                                                         math.cos(theta_avg), 0.0], [0.0, 0.0, 1.0]])
        self.dx = R @ dx_local
        self.x += self.dx * CONTROL_PERIOD

    # TODO: Make this available to the Vehicle class
    def get_encoder_offsets(self):
        offsets = []
        for caster in self.casters:
            caster.cancoder.configurator.refresh(caster.cancoder_cfg)  # Read current config
            curr_offset = caster.cancoder_cfg.magnet_sensor.magnet_offset
            caster.steer_position_signal.wait_for_update(0.1)
            curr_position = caster.cancoder.get_absolute_position().value
            offsets.append(f'{round(4096 * (curr_offset - curr_position))}.0 / 4096')
        print(f'ENCODER_MAGNET_OFFSETS = [{", ".join(offsets)}]')
