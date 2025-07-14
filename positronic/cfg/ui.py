# Configuration for the UI

import asyncio
import time
from collections import deque
from typing import List

import numpy as np

import configuronic as cfgc
import geom
import ironic as ir
import positronic.teleop


@cfgc.config(port=5005)
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


@cfgc.config(webxr=webxr_right,
             operator_position=positronic.teleop.FRANKA_FRONT_TRANSFORM,
             stream_to_webxr='first.image')
def teleop(webxr: ir.ControlSystem, operator_position: geom.Transform3D, stream_to_webxr: str | None = None):
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


@cfgc.config(webxr=webxr_both)
def teleop_umi(webxr: ir.ControlSystem, stream_to_webxr: str | None = None):
    teleop_cs = positronic.teleop.TeleopButtons()
    components = [webxr, teleop_cs]

    teleop_cs.bind(teleop_buttons=ir.utils.map_port(lambda x: x['right'], webxr.outs.buttons), )

    inputs = {'images': None}

    if stream_to_webxr:
        get_frame_for_webxr = ir.utils.MapPortCS(lambda frame: frame[stream_to_webxr])
        components.append(get_frame_for_webxr)
        inputs['images'] = (get_frame_for_webxr, 'input')
        webxr.bind(frame=get_frame_for_webxr.outs.output)

    outputs = {
        **teleop_cs.output_mappings,
        'controller_positions': webxr.outs.controller_positions,
    }

    return ir.compose(*components, inputs=inputs, outputs=outputs)


@cfgc.config(translation_speed=0.0005, rotation_speed=0.001, translation_dead_zone=0.8, rotation_dead_zone=0.7)
def spacemouse(translation_speed: float, rotation_speed: float, translation_dead_zone: float,
               rotation_dead_zone: float):
    from positronic.drivers.spacemouse import SpacemouseCS
    smouse = SpacemouseCS(translation_speed, rotation_speed, translation_dead_zone, rotation_dead_zone)
    inputs = {'robot_position': (smouse, 'robot_position'), 'robot_grip': None, 'images': None, 'robot_status': None}
    outputs = smouse.output_mappings
    outputs['metadata'] = ir.utils.const_property({'ui': 'spacemouse'})

    return ir.compose(smouse, inputs=inputs, outputs=outputs)


@cfgc.config(tracking=teleop, extra_ui_camera_names=['handcam_back', 'handcam_front', 'front_view', 'back_view'])
def teleop_with_ui(tracking: ir.ControlSystem, extra_ui_camera_names: List[str] | None):
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


@cfgc.config(camera_names=['handcam_left', 'handcam_right'])
def dearpygui_ui(camera_names: List[str]):
    from positronic.simulator.mujoco.mujoco_gui import DearpyguiUi
    return DearpyguiUi(camera_names)


@cfgc.config(joystick_id=0, fps=200, deadzone_size=0.1)
def gamepad(joystick_id, fps, deadzone_size):
    from positronic.drivers.ui.gamepad import GamepadCS
    return GamepadCS(joystick_id=joystick_id, fps=fps, deadzone_size=deadzone_size)


@cfgc.config(time_len_sec=5.0)
def stub(time_len_sec: float):  # noqa: C901  Function is too complex

    @ir.ironic_system(
        input_props=["robot_position"],
        output_ports=["robot_target_position", "gripper_target_grasp", "start_recording", "stop_recording", "reset"],
        output_props=["metadata"])
    class StubUi(ir.ControlSystem):
        """A stub UI that replays a pre-recorded trajectory.
        Used for testing and debugging purposes."""

        def __init__(self, time_len_sec: float = 5.0):
            super().__init__()
            self.events = deque()
            self.start_pos = None
            self.start_time = None
            self.time_len_sec = time_len_sec

        async def _start_recording(self, _):
            self.start_pos = (await self.ins.robot_position()).data
            await self.outs.start_recording.write(ir.Message(None))

        async def _send_target(self, time_sec):
            w = time_sec * (2 * np.pi) / 10
            delta = np.array([0, np.sin(w), np.cos(w)])
            translation = self.start_pos.translation + delta * 0.1
            rotation = self.start_pos.rotation
            await asyncio.gather(
                self.outs.robot_target_position.write(ir.Message(geom.Transform3D(translation, rotation))),
                self.outs.gripper_target_grasp.write(ir.Message(0.0)))

        async def _stop_recording(self, _):
            await self.outs.stop_recording.write(ir.Message(None))

        async def _reset(self, _):
            await self.outs.reset.write(ir.Message(None))

        async def setup(self):
            self.events.append((0.0, self._reset))
            time_sec = 3.0
            self.events.append((time_sec, self._start_recording))
            while time_sec < self.time_len_sec:
                self.events.append((time_sec, self._send_target))
                time_sec += 0.1
            self.events.append((time_sec, self._stop_recording))

            time_sec += 0.5
            self.events.append((time_sec, self._reset))

            time_sec += 2
            self.events.append((time_sec, None))  # Wait for reset to finish

            self.start_time = time.monotonic()

        async def step(self):
            current_time = time.monotonic() - self.start_time

            while self.events and self.events[0][0] < current_time:
                time_sec, callback = self.events.popleft()
                if callback is not None:
                    await callback(time_sec)

            return ir.State.FINISHED if not self.events else ir.State.ALIVE

        @ir.out_property
        async def metadata(self):
            return ir.Message({'ui': 'stub'})

    res = StubUi()
    inputs = {'robot_position': (res, 'robot_position'), 'images': None, 'robot_grip': None, 'robot_status': None}

    return ir.compose(res, inputs=inputs, outputs=res.output_mappings)
