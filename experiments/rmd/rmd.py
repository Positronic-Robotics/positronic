#!/usr/bin/env python3
"""
RMD V3 Motor Control - Automatic Sinusoidal Torque Control
Sends sinusoidal torque values to motors in sequence without user interaction
Uses CANUSBAdapter from usb_to_can_a.py for CAN communication.
"""

from copy import copy
import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Set

from usb_to_can_a import CANUSBAdapter, CAN_SPEED_MAP

import rerun as rr
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig


MYACTUATOR_TORQUE_CMD_ID = 0xA1
MYACTUATOR_DISCOVERY_CMD_ID = 0x9A

MYACTUATOR_MOTOR_REQ_ID_OFFSET = 0x140
MYACTUATOR_MOTOR_RESP_ID_OFFSET = 0x240

# Torque constants for different MyActuator models (in Nm/A)
TORQUE_CONSTANTS = {
    # X4-series actuators
    "X4V2": 0.625,
    "X4V3": 0.4,
    "X4_3": 0.75,
    "X4_24": 2.57,

    # X6-series actuators
    "X6V2": 0.875,
    "X6S2V2": 6.923,
    "X6V3": 1.25,
    "X6_7": 0.875,
    "X6_8": 1.25,
    "X6_40": 3.46,

    # X8-series actuators
    "X8V2": 2.093,
    "X8ProV2": 2.6,
    "X8S2V3": 7.8125,
    "X8HV3": 1.714,
    "X8ProHV3": 2.133,
    "X8_20": 1.923,
    "X8_25": 3.125,
    "X8_60": 7.5,
    "X8_90": 2.5,

    # X10-series actuators
    "X10V3": 2.264,
    "X10S2V3": 7.463,
    "X10_40": 2.308,
    "X10_100": 7.463,

    # X12-series actuators
    "X12_150": 3.333,

    # X15-series actuators
    "X15_400": 5.652,
}

@dataclass
class MotorState:
    temperature: int
    voltage: int
    error_state: int
    timestamp_ns: int
    position: int
    speed: int
    torque: int
    timestamp_ns: int



class FPSCounter:
    """Utility class for tracking and reporting frames per second (FPS).

    Counts frames and periodically reports the average FPS over the reporting interval.

    Args:
        prefix (str): Prefix string to use in FPS report messages
        report_every_sec (float): How often to report FPS, in seconds (default: 10.0)
    """

    def __init__(self, prefix: str, report_every_sec: float = 1.0):
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.reset()

    def reset(self):
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def report(self):
        fps = self.frame_count / (time.monotonic() - self.last_report_time)
        print(f"{self.prefix}: {fps:.2f} fps")
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()


class MotorController:
    """Abstraction for a single MyActuator motor on CAN bus."""
    def __init__(self, adapter: CANUSBAdapter, motor_id: int, model: str):
        self.adapter = adapter
        self.motor_id = motor_id
        self.state: Optional[MotorState] = None
        self._lock = threading.Lock()
        self.model = model
        self.read_fps = FPSCounter(f'motor/{self.motor_id}')
        self.write_fps = FPSCounter(f'motor/{self.motor_id}/send')

    def handle_feedback(self, can_id, data):
        # print(f"handle_feedback: {can_id}, {data}")
        if len(data) == 8 and data[0] == MYACTUATOR_TORQUE_CMD_ID and can_id == self.motor_id + MYACTUATOR_MOTOR_RESP_ID_OFFSET:
            temperature = data[1]
            torque = data[2] | (data[3] << 8)
            speed = data[4] | (data[5] << 8)
            position = data[6] | (data[7] << 8)

            speed = speed - 65536 if speed > 32767 else speed
            torque = torque - 65536 if torque > 32767 else torque
            position = position - 65536 if position > 32767 else position
            torque = torque / 100 * TORQUE_CONSTANTS[self.model]  # convert to Nm
            speed = math.radians(speed)
            position = math.radians(position)
            # position = position % 360  # TODO: comment this to get the raw position

            state = MotorState(
                temperature=temperature,
                voltage=0,
                error_state=0,
                timestamp_ns=time.time_ns(),
                position=position,
                speed=speed,
                torque=torque,
            )
            with self._lock:
                self.state = state
            self.read_fps.tick()
            print(f"[Feedback] Motor {self.motor_id}: pos={math.degrees(position):.1f}, speed={speed:.3f}, torque={torque:.3f}")

    def set_torque(self, torque: float):  # Torque is in Nm
        torque = int(torque / TORQUE_CONSTANTS[self.model] * 100)  # convert to centi-Amps
        torque = max(min(torque, 32767), -32768)
        payload = bytes([MYACTUATOR_TORQUE_CMD_ID, 0, 0, 0, torque & 0xFF, (torque >> 8) & 0xFF, 0, 0])
        self.adapter.send_can_frame(self.motor_id + MYACTUATOR_MOTOR_REQ_ID_OFFSET, payload)
        self.write_fps.tick()

    def get_state(self) -> Optional[MotorState]:
        with self._lock:
            return self.state

    def set_target_position(self, position: float):
        class LowPassFilter:
            """Simple low-pass filter implementation."""

            def __init__(self, alpha, initial_value):
                assert 0 < alpha <= 1, 'Alpha must be between 0 and 1'
                self.alpha = alpha
                self.y = initial_value

            def filter(self, x):
                self.y = self.alpha * x + (1 - self.alpha) * self.y
                return self.y

        K_p, K_d = 100, 20
        K_r, K_l, K_lp =  0.3, 75, 5 # 0.3, 75, 5  # 75
        K_r_inv = 1 / K_r
        K_r_K_l = K_r * K_l
        DT = 1 / 240  # 500  # 240

        q_s, dq_s = self.state.position, self.state.speed
        q_n, dq_n = q_s, dq_s
        q_d, dq_d = q_s, dq_s
        tau_filter = LowPassFilter(0.02, 0)

        otg = Ruckig(1, DT)
        otg_inp, otg_out = InputParameter(1), OutputParameter(1)
        otg_inp.max_velocity = np.array([4.0])
        otg_inp.max_acceleration = np.array([16.0])
        otg_inp.current_position = np.array([q_s])
        otg_inp.current_velocity = np.array([dq_s])
        otg_inp.target_position = np.array([position])
        otg_inp.target_velocity = np.array([0.0])

        otg_res = Result.Working

        while True:
            state = copy(self.get_state())
            q_s, dq_s = state.position, state.speed
            tau_s_f = tau_filter.filter(state.torque)

            if otg_res == Result.Working:
                otg_res = otg.update(otg_inp, otg_out)
                otg_out.pass_to_input(otg_inp)
                q_d = otg_out.new_position[0]
                dq_d = otg_out.new_velocity[0]

            tau_task = -K_p * (q_n - q_d) - K_d * (dq_n - dq_d)

            ddq_n = K_r_inv * (tau_task - tau_s_f)
            dq_n += ddq_n * DT
            q_n += dq_n * DT

            tau_f = K_r_K_l * ((dq_n - dq_s) + K_lp * (q_n - q_s))  # Nominal friction
            tau_n = tau_task + tau_f

            rr.set_time(f'motor_time', timestamp=np.datetime64(self.state.timestamp_ns, 'ns'))
            rr.set_time(f'time', timestamp=np.datetime64(time.time_ns(), 'ns'))
            rr.log(f'q/s', rr.Scalar(q_s))
            rr.log(f'q/n', rr.Scalar(q_n))
            rr.log(f'q/d', rr.Scalar(q_d))
            rr.log(f'dq/n', rr.Scalar(dq_n))
            rr.log(f'dq/d', rr.Scalar(dq_d))
            rr.log(f'dq/s', rr.Scalar(dq_s))
            rr.log(f'tau/task', rr.Scalar(tau_task))
            rr.log(f'tau/s_f', rr.Scalar(tau_s_f))
            rr.log(f'tau/f', rr.Scalar(tau_f))
            rr.log(f'tau/n', rr.Scalar(tau_n))
            rr.log(f'err/q', rr.Scalar(K_lp * (q_n - q_s)))
            rr.log(f'err/dq', rr.Scalar(K_r_K_l * (dq_n - dq_s)))

            self.set_torque(tau_n)
            time.sleep(0.002)


def discover_motors(adapter: CANUSBAdapter, id_range=range(1, 0x7F)):
    """Send 0x9A discovery command to all possible motor IDs and collect responses."""
    discovered: Dict[int, MotorState] = {}
    lock = threading.Lock()
    responses = {}

    def process_frame_hook(frame):
        can_id, data = orig_process_frame(frame)
        # 0x9A response: [0]=0x9A, [1]=temp, [3]=RlyCtrlRslt, [4-5]=voltage, [6-7]=errorState
        if len(data) == 8 and data[0] == MYACTUATOR_DISCOVERY_CMD_ID:
            temperature = data[1]
            voltage = data[4] | (data[5] << 8)
            error_state = data[6] | (data[7] << 8)
            with lock:
                responses[can_id] = MotorState(
                    temperature=temperature,
                    voltage=voltage,
                    error_state=error_state,
                    timestamp_ns=time.time_ns(),
                    position=0,
                    speed=0,
                    torque=0,
                )
            return
    orig_process_frame = adapter.process_frame
    adapter.process_frame = process_frame_hook

    payload = bytes([MYACTUATOR_DISCOVERY_CMD_ID, 0, 0, 0, 0, 0, 0, 0])
    for motor_id in id_range:
        adapter.send_can_frame(motor_id + MYACTUATOR_MOTOR_REQ_ID_OFFSET, payload)
        time.sleep(0.002)

    time.sleep(0.5)
    adapter.process_frame = orig_process_frame

    print("Discovered motors:")
    for mid, state in responses.items():
        print(f"  Motor ID {mid - MYACTUATOR_MOTOR_RESP_ID_OFFSET}: Temp={state.temperature}°C, Voltage={state.voltage}mV, Error=0x{state.error_state:04X}")

    return [mid - MYACTUATOR_MOTOR_RESP_ID_OFFSET for mid in responses.keys()]

def stop_motor(adapter: CANUSBAdapter, motor_id: int):
    """Send Motor Stop Command (0x81) to the given motor."""
    payload = bytes([0x81, 0, 0, 0, 0, 0, 0, 0])
    adapter.send_can_frame(motor_id + MYACTUATOR_MOTOR_REQ_ID_OFFSET, payload)


def angle_distance(a, b):
    """Calculate the minimal angle distance between two angles a and b."""
    diff = (b - a + 180) % 360 - 180
    return diff if diff != -180 else 180

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MyActuator RMD Motor Control (CANUSB)")
    parser.add_argument("-d", "--device", type=str, default="/dev/ttyUSB0", help="Serial device path (default: /dev/ttyUSB0)")
    parser.add_argument("-s", "--speed", type=int, choices=CAN_SPEED_MAP.keys(), default=1000000, help="CAN speed in bit/s (default: 1000000)")
    parser.add_argument("--amplitude", type=int, default=200, help="Sinusoidal amplitude (mNm)")
    parser.add_argument("--frequency", type=float, default=0.1, help="Sinusoidal frequency (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (s)")
    args = parser.parse_args()

    can_speed = CAN_SPEED_MAP[args.speed]
    adapter = CANUSBAdapter(args.device, can_speed=can_speed)

    # Step 1: Discover motors
    discovered_ids = discover_motors(adapter)
    if not discovered_ids:
        print("No motors discovered.")
        adapter.stop()
        return

    # discovered_ids = discovered_ids[2:3]
    # Step 2: Create controllers for each discovered motor
    discovered_ids = discovered_ids[:1]
    controllers = [MotorController(adapter, mid, "X10_100") for mid in discovered_ids]
    ctrl = controllers[0]

    # Step 3: Patch feedback for all controllers
    orig_process_frame = adapter.process_frame
    def process_frame_hook(frame):
        can_id, data = orig_process_frame(frame)
        ctrl.handle_feedback(can_id, data)
    adapter.process_frame = process_frame_hook

    rr.init('myactuator_motor_control1')
    rr.save('record.rrd')
    try:
        ctrl.set_torque(0)
        time.sleep(0.2)
        while True:
            print(f"Current position (deg): {math.degrees(ctrl.state.position)}")
            p = input("Enter target position: ")
            # p = 0
            target_position = math.radians(float(p))
            ctrl.set_target_position(target_position)

            ctrl.set_torque(0)
        print("Sinusoidal control finished. Sending stop command to all motors...")
        stop_motor(adapter, ctrl.motor_id)
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping motors...")
        ctrl.set_torque(0)
        stop_motor(adapter, ctrl.motor_id)
    finally:
        adapter.stop()

if __name__ == "__main__":
    main()
