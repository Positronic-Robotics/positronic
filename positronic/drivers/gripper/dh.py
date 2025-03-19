from ctypes import c_uint16
import time

import pymodbus.client as ModbusClient

import ironic as ir


# TODO: Migrate to async modbus
@ir.ironic_system(
    input_ports=['target_grip', 'force', 'speed'],
    output_props=['grip']
)
class DHGripper(ir.ControlSystem):
    def __init__(self, port: str):
        super().__init__()
        # TODO: Rewrite to use AsyncModbusSerialClient
        self.client = ModbusClient.ModbusSerialClient(
            port=port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
        )

    def _state_g(self):
        return self.client.read_holding_registers(0x200, count=1, slave=1).registers[0]

    def _state_r(self):
        return self.client.read_holding_registers(0x20A, count=1, slave=1).registers[0]

    async def setup(self):
        if self._state_g() != 1 or self._state_r() != 1:
            self.client.write_register(0x100, 0xa5, slave=1)
            while self._state_g() != 1 and self._state_r() != 1:
                time.sleep(0.1)

        # Set initial values
        await self.handle_force(ir.Message(data=100))  # Set to maximum force
        await self.handle_speed(ir.Message(data=100))  # Set to maximum speed
        await self.handle_target_grip(ir.Message(data=0))     # Open gripper
        time.sleep(0.5)

    @ir.on_message('target_grip')
    async def handle_target_grip(self, message: ir.Message):
        """Message data should be in range [0, 1]. 0 means fully open, 1 means fully closed."""
        width = round((1 - max(0, min(message.data, 1))) * 1000)
        self.client.write_register(0x103, c_uint16(width).value, slave=1)

    @ir.on_message('force')
    async def handle_force(self, message: ir.Message):
        self.client.write_register(0x101, c_uint16(message.data).value, slave=1)

    @ir.on_message('speed')
    async def handle_speed(self, message: ir.Message):
        self.client.write_register(0x104, c_uint16(message.data).value, slave=1)

    @ir.out_property
    async def grip(self):
        response = self.client.read_holding_registers(0x202, count=1, slave=1)
        if response.isError():
            raise Exception(f"Error reading gripper position: {response}")
        position = 1 - response.registers[0] / 1000
        return ir.Message(data=position)


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

    asyncio.run(_main())
