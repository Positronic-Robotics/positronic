from cfg import builds, store
import numpy as np

import ironic as ir


def get_camera_stub():
    @ir.ironic_system(
        output_ports=["frame"],
    )
    class StubCamera(ir.ControlSystem):
        def __init__(self):
            super().__init__()
            self.frame = ir.OutputPort.Stub()
            self.timestamp = 0

        async def step(self):
            frame_dict = {
                'image': np.zeros((32, 32, 3), dtype=np.uint8),
            }

            await self.outs.frame.write(ir.Message(frame_dict, timestamp=self.timestamp))
            self.timestamp += 1
            return ir.State.ALIVE

    return StubCamera()

camera_stub = builds(get_camera_stub)
camera_stub = store(camera_stub, name="camera_stub")
