import math
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generator, List

import can
import numpy as np

import ironic2 as ir


class CommandType(Enum):
    TORQUE = 0
    VELOCITY = 1
    POSITION = 2


@dataclass
class Command:
    type: CommandType
    value: np.array  # Value

    @staticmethod
    def position(value) -> 'Command':
        if isinstance(value, (int, float)):
            return Command(CommandType.POSITION, np.array([value]))
        else:
            return Command(CommandType.POSITION, np.array(value))

    @staticmethod
    def velocity(value) -> 'Command':
        if isinstance(value, (int, float)):
            return Command(CommandType.VELOCITY, np.array([value]))
        else:
            return Command(CommandType.VELOCITY, np.array(value))


@ABC
class MotorDriver:
    "Interface for vectorised motor drivers."
    # TODO: Now one driver manages ALL motors, though potentially we can extend to multiple
    # drivers on one bus. In this case the higher-level logic must be able to route incoming
    # messages to particular driver.

    @abstractmethod
    def decode(self, msg: can.Message, state: np.array) -> None:
        """Implementations must know how to map arbitration_id to motor index.

           Args:
             msg: CAN message that we received.
             state: Output np.array of (total_motors, 4) to write to:
                  4 stands for [timestamp, position, velocity, torque]
                    timestamp: UNIX epoch (seconds)
                    position:  Output shaft position in  radians, cla
                    velocity:  Output shaft velocity in rad/sec
                    torque:    Motor torque
        """
        pass

    @abstractmethod
    def encode(self, cmd: Command) -> Generator[can.Message]:
        pass

    @abstractmethod
    def ping(self) -> Generator[can.Message]:
        "Message to send to motor if there were no commands for some time"
        pass


class KTechMGMotorDriver(MotorDriver):

    @dataclass
    class MotorSpec:
        gear_ratio: float
        max_speed: float  # TODO: I still don't fully understand the dimension of this

    def __init__(self, motors: List[MotorSpec], start_can_id: int = 0x140):
        self._start_can_id = start_can_id
        self._num_motors = len(motors)
        self._specs = motors

    def decode(self, msg: can.Message, state: np.array) -> None:
        i = msg.arbitration_id - msg.arbitration_id
        if 0 <= i < self._num_motors:
            _, _, torque, vel, pos = struct.unpack('<bbhhH', msg.data)
            # TODO: Figure out the actual coeffcients
            state[0, i] = msg.timestamp
            state[1, i] = pos
            state[2, i] = vel
            state[3, i] = torque / 2048 * 33

    def encode(self, cmd: Command) -> Generator[can.Message]:
        match cmd.type:
            case CommandType.VELOCITY:
                for i, value in enumerate(cmd.value):
                    data = struct.pack('<Ii', 0xA2, int(value * self._specs[i] * 100))
                    yield can.Message(arbitration_id=self._start_can_id + i, data=data, is_extended_id=False)
            case CommandType.POSITION:
                for i, value in enumerate(cmd.value):
                    max_speed = int(self._specs[i].max_speed * self._specs[i].gear_ratio)
                    data = struct.pack('<HHi', 0xA4, max_speed,
                                       int(math.degrees(value) * self._specs[i].gear_ratio * 100))
                    yield can.Message(arbitration_id=self._start_can_id + i, data=data, is_extended_id=False)
            case CommandType.TORQUE:
                raise NotImplementedError('Torque control is not implemented yet for K-Tech motors')

    def ping(self) -> Generator[can.Message]:
        data = struct.pack('<II', 0x9C, 0x00)
        for i in range(self._num_motors):
            yield can.Message(arbitration_id=self._start_can_id + i, data=data, is_extended_id=False)


class CanBusDriver:
    commands = ir.SignalReader()  # Assume the data is Command
    state_writer = ir.SignalEmitter()  # The data is np.array of shape 3xN, where N is number of motors

    def __init__(self, bus_channel: str, num_motors: int, driver: MotorDriver, ping_every_sec: float) -> None:
        self._bus_channel = bus_channel
        self._driver = driver
        self._num_motors = num_motors
        self._ping_every_sec = ping_every_sec

    def run(self, should_stop: ir.SignalReader):
        commands = ir.ValueUpdated(self.commands)
        last_command_time = time.monotonic()

        state = np.zeros((4, self._num_motors), dtype=np.float32)  # Timestamp, position, velocity, torque

        with can.interface.Bus(channel=self._bus_channel, interface='socketcan') as bus:
            try:
                while not ir.signal_value(should_stop):
                    state_updated = False
                    while msg := bus.recv(0) is not None:
                        self._driver.decode(msg, state)
                        state_updated = True

                    if state_updated:
                        self.state_writer.emit(state[1:], np.max(state[0]))

                    cmd, cmd_updated = ir.signal_value(commands, (None, False))
                    if cmd_updated:
                        for msg in self._driver.encode(cmd):
                            bus.send(msg)
                        last_command_time = time.monotonic()
                    elif time.monotonic() - last_command_time > self._ping_every_sec:
                        for msg in self._driver.ping():
                            bus.send(msg)
                    else:
                        time.sleep(0.5 / 1000)  # We try to run 1kHz
            finally:
                cmd = Command.velocity(np.zeros(self._num_motors))
                for msg in self._driver.encode(cmd):
                    bus.send(msg)
                time.sleep(0.5)


if __name__ == "__main__":
    spec = KTechMGMotorDriver.MotorSpec(gear_ratio=3.6, max_speed=360 * 2)
    can_driver = CanBusDriver('can0', num_motors=1, driver=KTechMGMotorDriver([spec]), ping_every_sec=1 / 200)

    with ir.World() as world:
        command_writer, can_driver.commands = world.pipe()
        # TODO: Should be shared memory for performance
        can_driver.state_writer, state_reader = world.pipe()

        commands = [(0, Command.position(0)), (30., Command.position(2 * np.pi))]

        world.start(can_driver.run)
        i, start = 0, time.monotonic()
        while i < len(commands):
            t, cmd = commands[i]
            if time.monotonic() - start > t:
                i += 1
                command_writer.emit(cmd)
            else:
                # TODO: Probably should output the state
                time.sleep(0.01)
