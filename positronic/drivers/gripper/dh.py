import time
from collections.abc import Iterator
from ctypes import c_uint16
from typing import Any

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.utils import PendingMove, grip_setpoint, prepare_setpoint

with vendor_import('pymodbus', 'Gripper support'):
    import pymodbus.client as ModbusClient


class DHGripper(pimm.ControlSystem):
    _ARRIVED_TOL = 0.05  # the fingers report width, so arrival is judged from the reading

    def __init__(self, port: str, home_grip: float = 0.0):
        self.port = port
        self._home_grip = home_grip
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
        self.prepare = pimm.calls.ControlSystemHandler[Any, None](self)
        self.force = pimm.DefaultingReceiver(self, default=100)
        self.speed = pimm.DefaultingReceiver(self, default=100)

    @staticmethod
    def _initialize(client) -> Iterator[pimm.Sleep]:
        """Run the gripper's calibration, yielding until both axes report themselves ready."""

        def _state_g():
            return client.read_holding_registers(0x200, count=1, slave=1).registers[0]

        def _state_r():
            return client.read_holding_registers(0x20A, count=1, slave=1).registers[0]

        if _state_g() != 1 or _state_r() != 1:
            client.write_register(0x100, 0xA5, slave=1)
            while _state_g() != 1 and _state_r() != 1:
                yield pimm.Sleep(0.1)

    @staticmethod
    def _width(client) -> float:
        """How closed the fingers read back."""
        return 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000

    def _command(self, client, grip: float) -> None:
        """Put ``grip`` on the fingers, at the force and speed asked for."""
        client.write_register(0x103, c_uint16(round((1 - grip) * 1000)).value, slave=1)
        client.write_register(0x101, c_uint16(self.force.value).value, slave=1)
        client.write_register(0x104, c_uint16(self.speed.value).value, slave=1)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        client = ModbusClient.ModbusSerialClient(port=self.port, baudrate=115200, bytesize=8, parity='N', stopbits=1)
        client.connect()
        yield from self._initialize(client)

        last_grip = 0.0

        # TODO: Should we translate these to physical units (N and m/s)?
        try:
            with PendingMove[float](self._ARRIVED_TOL) as move:
                while not should_stop.value:
                    current_grip = self._width(client)
                    self.grip.emit(current_grip)

                    now = clock.now()
                    target = prepare_setpoint(move, self.prepare, self._home_grip, now)
                    if target is None:
                        target = grip_setpoint(move, self.sync_move, self.target_grip, current_grip, now)
                    if target is not None:
                        last_grip = target
                    self._command(client, last_grip)
                    move.answer()  # the width a settled move is answered with is on the fingers

                    yield pimm.Sleep(0.001)  # Small delay to prevent busy-waiting

                if move.active:  # the fingers push at the last width written to them, run or no run
                    self._command(client, self._width(client))
        finally:
            client.close()


if __name__ == '__main__':
    import numpy as np

    with pimm.World() as world:
        gripper = DHGripper('/dev/ttyUSB0')

        speed = world.pair(gripper.speed)
        force = world.pair(gripper.force)
        target_grip = world.pair(gripper.target_grip)
        grip = world.pair(gripper.grip)

        world.start([], background=gripper)

        print('Setting gripper to 20% speed and 100% force', flush=True)
        speed.emit(20)
        force.emit(100)

        for width in np.sin(np.linspace(0, 10 * np.pi, 60)) + 1:
            target_grip.emit(width)
            time.sleep(0.5)
            try:
                print(f'Real grip position: {grip.value}')
            except pimm.NoValueException:
                pass
