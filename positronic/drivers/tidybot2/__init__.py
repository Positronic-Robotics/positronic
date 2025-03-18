import asyncio
import ironic as ir
from positronic.drivers.tidybot2.base_controller import Vehicle


@ir.ironic_system(input_ports=['target_velocity_local', 'target_velocity_global', 'target_position'],
                  output_props=['position', 'velocity'])
class Tidybot(ir.ControlSystem):
    """Main control system interface for the Tidybot mobile base."""

    def __init__(self, encoder_magnet_offsets, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        super().__init__()
        self.vehicle = None
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.encoder_magnet_offsets = encoder_magnet_offsets

    async def setup(self):
        """Initialize the vehicle controller."""
        self.vehicle = Vehicle(encoder_magnet_offsets=self.encoder_magnet_offsets,
                               max_vel=self.max_vel,
                               max_accel=self.max_accel)
        self.vehicle.start_control()

    async def cleanup(self):
        """Clean up resources when shutting down."""
        if self.vehicle:
            self.vehicle.stop_control()
            self.vehicle = None

    @ir.on_message('target_velocity_local')
    async def handle_target_velocity_local(self, message: ir.Message):
        assert message.data.shape == (3, )
        self.vehicle.set_target_velocity(message.data, frame='local')

    @ir.on_message('target_velocity_global')
    async def handle_target_velocity_global(self, message: ir.Message):
        assert message.data.shape == (3, )
        self.vehicle.set_target_velocity(message.data, frame='global')

    @ir.on_message('target_position')
    async def handle_target_position(self, message: ir.Message):
        assert message.data.shape == (3, )
        self.vehicle.set_target_position(message.data)

    @ir.out_property
    async def position(self):
        return ir.Message(self.vehicle.x) if self.vehicle else ir.NoValue

    @ir.out_property
    async def velocity(self):
        return ir.Message(self.vehicle.dx) if self.vehicle else ir.NoValue
