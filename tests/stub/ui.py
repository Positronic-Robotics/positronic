from cfg import builds, store
import ironic as ir
import numpy as np
import time
from collections import deque
import asyncio
import geom

@ir.ironic_system(
    input_props=["robot_position"],
    output_ports=["robot_target_position", "gripper_target_grasp", "start_recording", "stop_recording", "reset"],
    output_props=["metadata"])
class StubUi(ir.ControlSystem):
    """A stub UI that replays a pre-recorded trajectory.
    Used for testing and debugging purposes."""

    def __init__(self):
        super().__init__()
        self.events = deque()
        self.start_pos = None
        self.start_time = None
        self.timestamp = 0

    async def _start_recording(self, _):
        self.start_pos = (await self.ins.robot_position()).data
        await self.outs.start_recording.write(ir.Message(None, timestamp=self.timestamp))

    async def _send_target(self, time_sec):
        translation = self.start_pos.translation + np.array([0, 0.1, 0.1]) * np.sin(time_sec * (2 * np.pi) / 3)
        quaternion = self.start_pos.quaternion
        await asyncio.gather(
            self.outs.robot_target_position.write(
                ir.Message(geom.Transform3D(translation, quaternion), timestamp=self.timestamp)),
            self.outs.gripper_target_grasp.write(ir.Message(0.0, timestamp=self.timestamp)))

    async def _stop_recording(self, _):
        await self.outs.stop_recording.write(ir.Message(None))

    async def setup(self):
        time_steps = 1
        self.events.append((time_steps, self._start_recording))
        while time_steps < 10:
            self.events.append((time_steps, self._send_target))
            time_steps += 2
        self.events.append((time_steps, self._stop_recording))

        self.start_time = time.monotonic()

    async def step(self):
        while self.events and self.events[0][0] < self.timestamp:
            time_sec, callback = self.events.popleft()
            await callback(time_sec)
        self.timestamp += 1

        return ir.State.FINISHED if not self.events else ir.State.ALIVE

    @ir.out_property
    async def metadata(self):
        return ir.Message({'ui': 'stub'})

def get_ui_stub():
    res = StubUi()
    inputs = {'robot_position': (res, 'robot_position'), 'images': None, 'robot_grip': None, 'robot_status': None}

    return ir.compose(res, inputs=inputs, outputs=res.output_mappings)

ui_stub = builds(get_ui_stub)
ui_stub = store(ui_stub, name="ui_stub")
