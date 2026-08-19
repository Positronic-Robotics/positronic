import time
from collections.abc import Iterator
from ctypes import c_uint16

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.arrival import PendingMove

with vendor_import('pymodbus', 'Gripper support'):
    import pymodbus.client as ModbusClient


_ARRIVED_TOL = 0.05  # the fingers report width, so arrival is judged from the reading


class DHGripper(pimm.ControlSystem):
    def __init__(self, port: str):
        self.port = port
        self.grip = pimm.ControlSystemEmitter[float](self)
        # Where the caller wants the grip to be
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        # The synchronous version of the above
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
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

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        client = ModbusClient.ModbusSerialClient(port=self.port, baudrate=115200, bytesize=8, parity='N', stopbits=1)
        client.connect()
        yield from self._initialize(client)

        last_grip = 0.0
        move = PendingMove(_ARRIVED_TOL)

        # TODO: Should we translate these to physical units (N and m/s)?
        while not should_stop.value:
            if not move.active:
                # A call first, and the stream only when none came: reading both would consume a streamed
                # target the call then overwrites, and a signal holds only its latest value.
                if (call := next(self.sync_move.incoming(), None)) is not None:
                    last_grip = float(call.request)
                    move.accept(call, last_grip, clock.now())
                elif (grip := pimm.value_updated(self.target_grip)) is not None:
                    last_grip = grip
            width = round((1 - max(0, min(last_grip, 1))) * 1000)
            client.write_register(0x103, c_uint16(width).value, slave=1)
            client.write_register(0x101, c_uint16(self.force.value).value, slave=1)
            client.write_register(0x104, c_uint16(self.speed.value).value, slave=1)

            current_grip = 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000
            self.grip.emit(current_grip)
            if move.active:
                # A grasp is a move that times out: the fingers stop on the object. The target stands, so
                # they keep holding it.
                move.settle(current_grip, clock.now())

            yield pimm.Sleep(0.001)  # Small delay to prevent busy-waiting

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
