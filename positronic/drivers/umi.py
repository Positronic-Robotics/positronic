import ironic as ir
import geom

# registration transform for the last gripper
# TODO: remove in the next PR
GRIPPER_REGISTRATION = geom.Rotation.from_quat([-0.08998721, -0.29523472, -0.51761315,  0.79800714])


@ir.ironic_system(
    input_ports=['tracker_position', 'target_grip'],
    output_props=[
        'ee_position',
        'grip',
        'metadata',
        'umi_left_quaternion',
        'umi_left_translation',
        'umi_right_quaternion',
        'umi_right_translation',
    ],
)
class UmiCS(ir.ControlSystem):

    def __init__(self):
        super().__init__()
        self.tracker_positions = {'left': None, 'right': None}
        self.prev_tracker_positions = {'left': None, 'right': None}
        self.target_grip = None
        self.relative_gripper_transform = None

    @ir.out_property
    async def ee_position(self):
        if (self.tracker_positions['left'] is None or self.tracker_positions['right'] is None or
            self.prev_tracker_positions['left'] is None or self.prev_tracker_positions['right'] is None):
            return ir.Message(data=ir.NoValue, timestamp=ir.system_clock())

        left_pos = self.tracker_positions['left']
        right_pos = self.tracker_positions['right']

        if self.relative_gripper_transform is None:
            self.relative_gripper_transform = left_pos.inv * right_pos

        relative_gripper_transform = self.relative_gripper_transform.inv * (self.prev_tracker_positions['right'].inv * self.tracker_positions['right']) * self.relative_gripper_transform

        return ir.Message(data=relative_gripper_transform, timestamp=ir.system_clock())

    @ir.out_property
    async def umi_left_quaternion(self):
        return ir.Message(data=self.tracker_positions['left'].rotation.as_quat.copy(), timestamp=ir.system_clock())

    @ir.out_property
    async def umi_left_translation(self):
        return ir.Message(data=self.tracker_positions['left'].translation.copy(), timestamp=ir.system_clock())

    @ir.out_property
    async def umi_right_quaternion(self):
        return ir.Message(data=self.tracker_positions['right'].rotation.as_quat.copy(), timestamp=ir.system_clock())

    @ir.out_property
    async def umi_right_translation(self):
        return ir.Message(data=self.tracker_positions['right'].translation.copy(), timestamp=ir.system_clock())
    @ir.out_property
    async def grip(self):
        return ir.Message(data=self.target_grip, timestamp=ir.system_clock())

    @ir.out_property
    async def metadata(self):
        return ir.Message(data={'source': 'umi'})

    @ir.on_message('tracker_position')
    async def on_tracker_positions(self, message: ir.Message):
        if message.data['left'] is None or message.data['right'] is None:
            return

        if self.tracker_positions['left'] is None or self.tracker_positions['right'] is None:
            self.tracker_positions = message.data
        else:
            self.prev_tracker_positions = {'left': self.tracker_positions['left'].copy(), 'right': self.tracker_positions['right'].copy()}
            self.tracker_positions = message.data

    @ir.on_message('target_grip')
    async def on_target_grip(self, message: ir.Message):
        self.target_grip = message.data
