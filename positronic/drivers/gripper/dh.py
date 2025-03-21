from ctypes import c_uint16
import time
import multiprocessing as mp

import pymodbus.client as ModbusClient

import ironic as ir


def _gripper_process(
        port: str,
        target_grip: mp.Value,
        current_grip: mp.Value,
        force: mp.Value,
        speed: mp.Value,
        running: mp.Value
):
    client = ModbusClient.ModbusSerialClient(
        port=port,
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

    while running.value:
        # Update gripper based on shared values
        width = round((1 - max(0, min(target_grip.value, 1))) * 1000)
        client.write_register(0x103, c_uint16(width).value, slave=1)
        client.write_register(0x101, c_uint16(force.value).value, slave=1)
        client.write_register(0x104, c_uint16(speed.value).value, slave=1)
        current_grip.value = 1 - client.read_holding_registers(0x202, count=1, slave=1).registers[0] / 1000
        time.sleep(0.001)  # Small delay to prevent busy-waiting

    client.close()

# TODO: Migrate to async modbus
@ir.ironic_system(
    input_ports=['target_grip', 'force', 'speed'],
    output_props=['grip']
)
class DHGripper(ir.ControlSystem):
    def __init__(self, port: str):
        super().__init__()
        self.running = mp.Value('b', True)
        self.target_grip = mp.Value('f', 0)
        self.current_grip = mp.Value('f', 0)
        self.force = mp.Value('i', 100)
        self.speed = mp.Value('i', 100)
        self.process = mp.Process(
            target=_gripper_process,
            args=(port, self.target_grip, self.current_grip, self.force, self.speed, self.running)
        )

    async def setup(self):
        self.process.start()

    @ir.on_message('target_grip')
    async def handle_target_grip(self, message: ir.Message):
        """Message data should be in range [0, 1]. 0 means fully open, 1 means fully closed."""
        self.target_grip.value = message.data

    @ir.on_message('force')
    async def handle_force(self, message: ir.Message):
        self.force.value = message.data

    @ir.on_message('speed')
    async def handle_speed(self, message: ir.Message):
        self.speed.value = message.data

    @ir.out_property
    async def grip(self):
        return ir.Message(data=self.current_grip.value)

    async def cleanup(self):
        self.running.value = False
        self.process.join()


if __name__ == "__main__":
    import asyncio
    import numpy as np

    async def _main():
        gripper = DHGripper("/dev/ttyUSB0")
        grip_port = ir.OutputPort('target_grip')
        gripper.bind(target_grip=grip_port)

        await gripper.setup()
        await gripper.handle_speed(ir.Message(data=20))
        await gripper.handle_force(ir.Message(data=100))

        for width in (np.sin(np.linspace(0, 10 * np.pi, 60)) + 1):
            await grip_port.write(ir.Message(data=width))
            await asyncio.sleep(0.25)
            msg = await gripper.grip()
            print(f"Real grip position: {msg.data}")

        await gripper.cleanup()

    asyncio.run(_main())
