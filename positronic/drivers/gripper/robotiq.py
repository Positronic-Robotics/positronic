"""Robotiq 2F-85 (and 2F-140) Modbus RTU driver (RS-485)."""

from collections.abc import Iterator

import pymodbus.client as ModbusClient

import pimm

_REG_CMD = 0x03E8
_REG_IN_POS = 0x07D2
_SLAVE = 9
_BAUD_RATE = 115200
_BYTESIZE = 8
_PARITY = 'N'
_STOPBITS = 1


class Robotiq2F(pimm.ControlSystem):
    def __init__(self, port: str):
        self._port = port
        self.grip = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver(self)
        self.force = pimm.ControlSystemReceiver(self)
        self.speed = pimm.ControlSystemReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        client = ModbusClient.ModbusSerialClient(
            port=self._port, baudrate=_BAUD_RATE, bytesize=_BYTESIZE, parity=_PARITY, stopbits=_STOPBITS
        )
        assert client.connect(), f'Failed to connect to Robotiq gripper at {self._port}'

        try:
            limiter = pimm.RateLimiter(clock, hz=200)  # According to the manual, the gripper can handle 200Hz
            client.write_registers(_REG_CMD, [0x0000, 0x0000, 0x0000], device_id=_SLAVE)
            client.write_registers(_REG_CMD, [0x0100, 0x0000, 0x0000], device_id=_SLAVE)

            target_grip = pimm.DefaultReceiver(pimm.ValueUpdated(self.target_grip), (None, False))
            force = pimm.DefaultReceiver(self.force, 255)  # device scale 0..255
            speed = pimm.DefaultReceiver(self.speed, 255)  # device scale 0..255

            while not should_stop.value:
                pos_val, updated = target_grip.value
                if updated:
                    pos = int(max(0, min(1, pos_val)) * 255)
                    spd = int(max(0, min(255, speed.value)))
                    frc = int(max(0, min(255, force.value)))

                    client.write_registers(_REG_CMD, [0x0900, pos, (frc << 8) | spd], device_id=_SLAVE)

                reg = client.read_input_registers(_REG_IN_POS, count=1, device_id=_SLAVE).registers[0]
                self.grip.emit(min(1.0, max(0.0, (reg >> 8) / 255.0)))

                yield pimm.Sleep(limiter.wait_time())
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
