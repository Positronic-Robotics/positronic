import numpy as np

import ironic as ir
import geom

@ir.ironic_system(
    input_ports=['tracker_position', 'target_grip'],
    output_props=['ee_position', 'grip', 'metadata'],
)
class UmiCS(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.tracker_position = geom.Transform3D()
        self.target_grip = None

    @ir.out_property
    async def ee_position(self):
        # TODO: here we will put registered robot position
        return ir.Message(data=self.tracker_position, timestamp=ir.system_clock())

    @ir.out_property
    async def grip(self):
        return ir.Message(data=self.target_grip, timestamp=ir.system_clock())

    @ir.out_property
    async def metadata(self):
        return ir.Message(data={})

    @ir.on_message('tracker_position')
    async def on_tracker_position(self, message: ir.Message):
        self.tracker_position = message.data

    @ir.on_message('target_grip')
    async def on_target_grip(self, message: ir.Message):
        self.target_grip = message.data
