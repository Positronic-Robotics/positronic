# Configuration for the UI

import asyncio
from collections import deque
import time
from typing import List, Optional

import numpy as np

import geom
import ironic as ir
import positronic.teleop


@ir.config(port=5005)
def webxr(port: int, require_left: bool, require_right: bool):
    from positronic.webxr import WebXR

    def _filter_none(data: dict):
        if require_left and data['left'] is None:
            return ir.NoValue
        if require_right and data['right'] is None:
            return ir.NoValue
        return data

    webxr = WebXR(port=port)
    webxr = ir.extend(webxr,
                      controller_positions=ir.utils.map_port(_filter_none, webxr.outs.controller_positions),
                      buttons=ir.utils.map_port(_filter_none, webxr.outs.buttons))

    return webxr


webxr_both = webxr.override(require_left=True, require_right=True)
webxr_left = webxr.override(require_left=True, require_right=False)
webxr_right = webxr.override(require_left=False, require_right=True)


@ir.config(webxr=webxr_right, operator_position=positronic.teleop.FRANKA_FRONT_TRANSFORM, stream_to_webxr='first.image')
def teleop(webxr: ir.ControlSystem,
           operator_position: geom.Transform3D,
           stream_to_webxr: Optional[str] = None):
    teleop_cs = positronic.teleop.TeleopSystem(operator_position)
    components = [webxr, teleop_cs]

    teleop_cs.bind(
        teleop_transform=ir.utils.map_port(lambda x: x['right'], webxr.outs.controller_positions),
        teleop_buttons=ir.utils.map_port(lambda x: x['right'], webxr.outs.buttons),
    )

    inputs = {'robot_position': (teleop_cs, 'robot_position'), 'images': None, 'robot_grip': None, 'robot_status': None}

    if stream_to_webxr:
        get_frame_for_webxr = ir.utils.MapPortCS(lambda frame: frame[stream_to_webxr])
        components.append(get_frame_for_webxr)
        inputs['images'] = (get_frame_for_webxr, 'input')
        webxr.bind(frame=get_frame_for_webxr.outs.output)

    return ir.compose(*components, inputs=inputs, outputs=teleop_cs.output_mappings)


@ir.config(translation_speed=0.0005, rotation_speed=0.001, translation_dead_zone=0.8, rotation_dead_zone=0.7)
def spacemouse(translation_speed: float, rotation_speed: float, translation_dead_zone: float,
               rotation_dead_zone: float):
    from positronic.drivers.spacemouse import SpacemouseCS
    smouse = SpacemouseCS(translation_speed, rotation_speed, translation_dead_zone, rotation_dead_zone)
    inputs = {'robot_position': (smouse, 'robot_position'), 'robot_grip': None, 'images': None, 'robot_status': None}
    outputs = smouse.output_mappings
    outputs['metadata'] = ir.utils.const_property({'ui': 'spacemouse'})

    return ir.compose(smouse, inputs=inputs, outputs=outputs)


@ir.config(tracking=teleop, extra_ui_camera_names=['handcam_back', 'handcam_front', 'front_view', 'back_view'])
def teleop_with_ui(tracking: ir.ControlSystem, extra_ui_camera_names: Optional[List[str]]):
    if extra_ui_camera_names:
        from positronic.simulator.mujoco.mujoco_gui import DearpyguiUi
        gui = DearpyguiUi(extra_ui_camera_names)
        components = [tracking, gui]

        inputs = {
            'robot_position': [(tracking, 'robot_position'), (gui, 'robot_position')],
            'images': (gui, 'images'),
            'robot_grip': (gui, 'robot_grip'),
            'robot_status': (gui, 'robot_status')
        }
        outputs = tracking.output_mappings
        return ir.compose(*components, inputs=inputs, outputs=outputs)

    return tracking


@ir.config(camera_names=['handcam_left', 'handcam_right'])
def dearpygui_ui(camera_names: List[str]):
    from positronic.simulator.mujoco.mujoco_gui import DearpyguiUi
    return DearpyguiUi(camera_names)


@ir.config
def stub():

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

        async def _start_recording(self, _):
            self.start_pos = (await self.ins.robot_position()).data
            await self.outs.start_recording.write(ir.Message(None))

        async def _send_target(self, time_sec):
            translation = self.start_pos.translation + np.array([0, 0.1, 0.1]) * np.sin(time_sec * (2 * np.pi) / 10)
            rotation = self.start_pos.rotation
            await asyncio.gather(
                self.outs.robot_target_position.write(ir.Message(geom.Transform3D(translation, rotation))),
                self.outs.gripper_target_grasp.write(ir.Message(0.0)))

        async def _stop_recording(self, _):
            await self.outs.stop_recording.write(ir.Message(None))

        async def setup(self):
            time_sec = 0.1
            self.events.append((time_sec, self._start_recording))
            while time_sec < 5:
                self.events.append((time_sec, self._send_target))
                time_sec += 0.1
            self.events.append((time_sec, self._stop_recording))

            self.start_time = time.monotonic()

        async def step(self):
            current_time = time.monotonic() - self.start_time

            while self.events and self.events[0][0] < current_time:
                time_sec, callback = self.events.popleft()
                await callback(time_sec)

            return ir.State.FINISHED if not self.events else ir.State.ALIVE

        @ir.out_property
        async def metadata(self):
            return ir.Message({'ui': 'stub'})

    res = StubUi()
    inputs = {'robot_position': (res, 'robot_position'), 'images': None, 'robot_grip': None, 'robot_status': None}

    return ir.compose(res, inputs=inputs, outputs=res.output_mappings)
