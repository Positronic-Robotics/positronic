from ctypes import c_uint16
import time

import pymodbus.client as ModbusClient

import ironic2 as ir


class DHGripper(ir.ControlSystem):
    def __init__(self, comms: ir.CommunicationProvider, port: str):
        self.port = port
        self.target_grip = comms.reader('target_grip')
        self.force = comms.reader('force')
        self.speed = comms.reader('speed')
        self.grip = comms.emitter('grip')

        self.should_stop = comms.should_stop()

    def run(self):
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

        while ir.signal_is_true(self.should_stop):
            # Update gripper based on shared values
            target_grip = ir.signal_value(self.target_grip, 0)
            width = round((1 - max(0, min(target_grip, 1))) * 1000)
            client.write_register(0x103, c_uint16(width).value, slave=1)
            client.write_register(0x101, c_uint16(ir.signal_value(self.force, 100)).value, slave=1)
            client.write_register(0x104, c_uint16(ir.signal_value(self.speed, 100)).value, slave=1)

            current_grip = 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000
            self.grip.emit(ir.Message(current_grip))
            time.sleep(0.001)  # Small delay to prevent busy-waiting

        client.close()


if __name__ == "__main__":
    import numpy as np

    world = ir.mp.MPWorld()
    gripper = world.add_background_control_system(DHGripper, "/dev/ttyUSB0")

    def main_loop(should_stop: ir.SignalReader):
        gripper.speed.emit(ir.Message(data=20))
        gripper.force.emit(ir.Message(data=100))

        for width in (np.sin(np.linspace(0, 10 * np.pi, 60)) + 1):
            gripper.target_grip.emit(ir.Message(data=width))
            time.sleep(0.25)
            print(f"Real grip position: {ir.signal_value(gripper.grip, 0)}")

    world.run(main_loop)
