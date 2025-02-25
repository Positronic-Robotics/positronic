# Configuration for the UI

from typing import List, Optional
from hydra_zen import builds, store

import ironic as ir

def _webxr(port: int):
    from webxr import WebXR

    return WebXR(port=port)
    components = [webxr]
    inputs, outputs = {}, {'transform': webxr.outs.transform, 'buttons': webxr.outs.buttons}

    if stream:
        get_frame_for_webxr = ir.utils.MapPortCS(lambda frame: frame[stream])
        components.append(get_frame_for_webxr)
        inputs['images'] = (get_frame_for_webxr, 'input')
        webxr.bind(frame=get_frame_for_webxr.outs.output)
    else:
        inputs['images'] = None

    return ir.compose(*components, inputs=inputs, outputs=outputs)


def _teleop(webxr: ir.ControlSystem, operator_position: str, stream_to_webxr: Optional[str] = None, pos_transform: str = 'teleop'):
    from teleop import TeleopSystem
    # TODO: Support Andrei's transformation (aka 'pos_transform')
    teleop = TeleopSystem(operator_position=operator_position)
    components = [webxr, teleop]

    teleop.bind(teleop_transform=webxr.outs.transform,
                teleop_buttons=webxr.outs.buttons)

    inputs = {'robot_position': (teleop, 'robot_position'),
              'images': None,
              'robot_grip': None,
              'robot_status': None}

    if stream_to_webxr:
        get_frame_for_webxr = ir.utils.MapPortCS(lambda frame: frame[stream_to_webxr])
        components.append(get_frame_for_webxr)
        inputs['images'] = (get_frame_for_webxr, 'input')
        webxr.bind(frame=get_frame_for_webxr.outs.output)

    return ir.compose(*components, inputs=inputs, outputs=teleop.output_mappings)


def _spacemouse(translation_speed: float = 0.0005,
                rotation_speed: float = 0.001,
                translation_dead_zone: float = 0.8,
                rotation_dead_zone: float = 0.7):
    from drivers.spacemouse import SpacemouseCS
    smouse = SpacemouseCS(translation_speed, rotation_speed, translation_dead_zone, rotation_dead_zone)
    inputs = {'robot_position': (smouse, 'robot_position'),
              'robot_grip': None,
              'images': None,
              'robot_status': None}
    outputs = smouse.output_mappings
    outputs['metadata'] = None

    return ir.compose([smouse], inputs=inputs, outputs=outputs)


def _teleop_ui(tracking: ir.ControlSystem, extra_ui_camera_names: Optional[List[str]] = None):
    if extra_ui_camera_names:
        from simulator.mujoco.mujoco_gui import DearpyguiUi
        gui = DearpyguiUi(extra_ui_camera_names)
        components = [tracking, gui]

        inputs = {'robot_position': [(tracking, 'robot_position'), (gui, 'robot_position')],
                  'images': (gui, 'images'),
                  'robot_grip': (gui, 'robot_grip'),
                  'robot_status': (gui, 'robot_status')}
        outputs = tracking.output_mappings
        return ir.compose(*components, inputs=inputs, outputs=outputs)

    return tracking


def _dearpygui_ui(camera_names: List[str]):
    from simulator.mujoco.mujoco_gui import DearpyguiUi
    return DearpyguiUi(camera_names)


webxr = builds(_webxr, populate_full_signature=True)
teleop = builds(_teleop, populate_full_signature=True)
teleop_ui = builds(_teleop_ui, populate_full_signature=True)
spacemouse = builds(_spacemouse, populate_full_signature=True)
dearpygui_ui = builds(_dearpygui_ui, populate_full_signature=True)


ui_store = store(group="ui")
ui_store(teleop(webxr(port=5005), operator_position='back', stream_to_webxr='image'), name='teleop')
ui_store(teleop_ui(teleop(webxr(port=5005), operator_position='back'),
                extra_ui_camera_names=['handcam_back', 'handcam_front', 'front_view', 'back_view']),
      name='teleop_gui')

ui_store(dearpygui_ui(camera_names=['handcam_left', 'handcam_right']), name='gui')

ui_store.add_to_hydra_store()