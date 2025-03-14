import ironic as ir
import geom


@ir.ironic_system(
    input_ports=['tracker_position'],
    output_props=['metadata', 'umi_left', 'umi_right'],
)
class UmiCS(ir.ControlSystem):

    def __init__(self, registration: geom.Transform3D):
        super().__init__()
        self.tracker_positions = {'left': None, 'right': None}
        self.prev_tracker_positions = {'left': None, 'right': None}
        self.relative_gripper_transform = None
        self.registration = registration

    @ir.out_property
    async def umi_left(self):
        return ir.Message(data=self.tracker_positions['left'].copy(), timestamp=ir.system_clock())

    @ir.out_property
    async def umi_right(self):
        return ir.Message(data=self.tracker_positions['right'].copy(), timestamp=ir.system_clock())

    @ir.out_property
    async def metadata(self):
        return ir.Message(data={'source': 'umi', 'registration_transform': self.registration})

    @ir.on_message('tracker_position')
    async def on_tracker_positions(self, message: ir.Message):
        if self.tracker_positions['left'] is None or self.tracker_positions['right'] is None:
            self.tracker_positions = message.data
        else:
            self.prev_tracker_positions = {
                'left': self.tracker_positions['left'].copy(),
                'right': self.tracker_positions['right'].copy(),
            }
            self.tracker_positions = message.data
