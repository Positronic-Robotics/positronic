from ctypes import c_uint16
import time

import pymodbus.client as ModbusClient

import pimm


class DHGripper(pimm.ControlSystem):
    def __init__(self, port: str):
        self.port = port
        self.grip: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self.target_grip: pimm.SignalReceiver = pimm.ControlSystemReceiver(self)
        self.force: pimm.SignalReceiver = pimm.ControlSystemReceiver(self)
        self.speed: pimm.SignalReceiver = pimm.ControlSystemReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        client = ModbusClient.ModbusSerialClient(
            port=self.port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
        )
        client.connect()

        def _state_g():
            return client.read_holding_registers(0x200, count=1, slave=1).registers[0]

        def _state_r():
            return client.read_holding_registers(0x20A, count=1, slave=1).registers[0]

        # Initial setup
        if _state_g() != 1 or _state_r() != 1:
            client.write_register(0x100, 0xa5, slave=1)
            while _state_g() != 1 and _state_r() != 1:
                yield pimm.Sleep(0.1)

        target_grip = pimm.DefaultReceiver(self.target_grip, 0)
        # TODO: We must translate these to physical units (N and m/s)
        force = pimm.DefaultReceiver(self.force, 100)
        speed = pimm.DefaultReceiver(self.speed, 100)

        while not should_stop.value:
            # Update gripper based on shared values
            try:
                width = round((1 - max(0, min(target_grip.value, 1))) * 1000)
                client.write_register(0x103, c_uint16(width).value, slave=1)
                client.write_register(0x101, c_uint16(force.value).value, slave=1)
                client.write_register(0x104, c_uint16(speed.value).value, slave=1)
            except pimm.NoValueException:
                pass

            current_grip = 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000
            self.grip.emit(current_grip)

            yield pimm.Sleep(0.001)  # Small delay to prevent busy-waiting

        client.close()


if __name__ == "__main__":
    import numpy as np

    with pimm.World() as world:
        gripper = DHGripper("/dev/ttyUSB0")

        speed = world.pair(gripper.speed)
        force = world.pair(gripper.force)
        target_grip = world.pair(gripper.target_grip)
        grip = world.pair(gripper.grip)

        world.start([], background=gripper)

        print("Setting gripper to 20% speed and 100% force", flush=True)
        speed.emit(20)
        force.emit(100)

        for width in (np.sin(np.linspace(0, 10 * np.pi, 60)) + 1):
            target_grip.emit(width)
            time.sleep(0.5)
            try:
                print(f"Real grip position: {grip.value}")
            except pimm.NoValueException:
                pass
