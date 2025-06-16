from ctypes import c_uint16
import time

import pymodbus.client as ModbusClient

import ironic2 as ir


class DHGripper:

    grip: ir.SignalEmitter = ir.NoOpEmitter()
    target_grip: ir.SignalReader = ir.NoOpReader()
    force: ir.SignalReader = ir.NoOpReader()
    speed: ir.SignalReader = ir.NoOpReader()

    def __init__(self, port: str):
        self.port = port

    def run(self, should_stop: ir.SignalReader):
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
                time.sleep(0.1)

        # Set initial values
        client.write_register(0x101, c_uint16(100).value, slave=1)  # force
        client.write_register(0x104, c_uint16(100).value, slave=1)  # speed
        width = round((1 - 0) * 1000)  # fully open
        client.write_register(0x103, c_uint16(width).value, slave=1)
        time.sleep(0.5)

        while not ir.is_true(should_stop):
            # Update gripper based on shared values
            try:
                target_grip = ir.signal_value(self.target_grip)
                width = round((1 - max(0, min(target_grip, 1))) * 1000)
                client.write_register(0x103, c_uint16(width).value, slave=1)
                client.write_register(0x101, c_uint16(ir.signal_value(self.force)).value, slave=1)
                client.write_register(0x104, c_uint16(ir.signal_value(self.speed)).value, slave=1)

                current_grip = 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000
                self.grip.emit(current_grip)
            except ir.NoValueException:
                pass

            time.sleep(0.001)  # Small delay to prevent busy-waiting

        client.close()


if __name__ == "__main__":
    import numpy as np

    with ir.World() as world:
        gripper = DHGripper("/dev/ttyUSB0")

        speed, gripper.speed = world.pipe()
        force, gripper.force = world.pipe()
        target_grip, gripper.target_grip = world.pipe()
        gripper.grip, grip = world.pipe()

        world.start(gripper.run)

        print("Setting gripper to 20% speed and 100% force", flush=True)
        speed.emit(20)
        force.emit(100)

        for width in (np.sin(np.linspace(0, 10 * np.pi, 60)) + 1):
            target_grip.emit(width)
            time.sleep(0.5)
            try:
                print(f"Real grip position: {ir.signal_value(grip)}")
            except ir.NoValueException:
                pass
