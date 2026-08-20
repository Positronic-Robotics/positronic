"""Robotiq 2F-85 (and 2F-140) Modbus RTU driver (RS-485)."""

from collections.abc import Iterator

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.utils import PendingMove, grip_setpoint

with vendor_import('pymodbus', 'Gripper support'):
    import pymodbus.client as ModbusClient

_REG_CMD = 0x03E8
_REG_IN_POS = 0x07D2
_SLAVE = 9
_BAUD_RATE = 115200
_BYTESIZE = 8
_PARITY = 'N'
_STOPBITS = 1


class Robotiq2F(pimm.ControlSystem):
    _ARRIVED_TOL = 0.05  # the fingers report width, so arrival is judged from the reading

    def __init__(self, port: str):
        self._port = port
        self.grip = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
        self.force = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255
        self.speed = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        client = ModbusClient.ModbusSerialClient(
            port=self._port, baudrate=_BAUD_RATE, bytesize=_BYTESIZE, parity=_PARITY, stopbits=_STOPBITS
        )
        assert client.connect(), f'Failed to connect to Robotiq gripper at {self._port}'

        move = PendingMove(self._ARRIVED_TOL)
        try:
            limiter = pimm.RateLimiter(clock, hz=200)  # According to the manual, the gripper can handle 200Hz
            client.write_registers(_REG_CMD, [0x0000, 0x0000, 0x0000], device_id=_SLAVE)
            client.write_registers(_REG_CMD, [0x0100, 0x0000, 0x0000], device_id=_SLAVE)

            while not should_stop.value:
                reg = client.read_input_registers(_REG_IN_POS, count=1, device_id=_SLAVE).registers[0]
                grip = min(1.0, max(0.0, (reg >> 8) / 255.0))
                self.grip.emit(grip)

                target = grip_setpoint(move, self.sync_move, self.target_grip, grip, clock.now())
                if target is not None:
                    pos = int(target * 255)
                    spd = int(max(0, min(255, self.speed.value)))
                    frc = int(max(0, min(255, self.force.value)))

                    client.write_registers(_REG_CMD, [0x0900, pos, (frc << 8) | spd], device_id=_SLAVE)

                yield limiter.wait()
        except Exception as exc:
            move.fail(exc)  # a run that dies mid-move must not leave its asker waiting
            raise
        finally:
            client.close()


if __name__ == '__main__':
    import time

    with pimm.World() as world:
        gr = Robotiq2F(port='/dev/ttyUSB0')

        spd = world.pair(gr.speed)
        frc = world.pair(gr.force)
        tgt = world.pair(gr.target_grip)
        grip = world.pair(gr.grip)

        world.start([], background=gr)

        spd.emit(128)
        frc.emit(128)

        start = time.time()
        waypoints = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0]
        i = 0

        while True:
            if time.time() - start > i * 1.0:
                tgt.emit(waypoints[i])
                i += 1
                if i >= len(waypoints):
                    break
            time.sleep(0.1)
            try:
                print(f'[{i}] Grip: {grip.value:.2f}')
            except pimm.NoValueException:
                pass
