# Configuration for the robotics environment
from typing import Optional

import numpy as np

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.roboarms
import ironic as ir


def _state_mapping(env: ir.ControlSystem):
    robot_properties = {}
    for name, mapping in env.output_mappings.items():
        if name == 'metadata':
            continue
        if ir.is_property(mapping):
            robot_properties[name] = mapping

    return ir.extend(env, state=ir.utils.properties_dict(**robot_properties))


@ir.config(camera=None)
def roboarm(arm: ir.ControlSystem, camera: ir.ControlSystem | None = None):
    components = [arm]
    inputs = {'target_position': (arm, 'target_position'), 'reset': (arm, 'reset'), 'target_grip': (arm, 'target_grip')}

    outputs = {**arm.output_mappings, 'frame': ir.OutputPort.Stub()}
    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    if 'ext_force_ee' not in arm.output_mappings:
        outputs['ext_force_ee'] = ir.utils.const_property(np.zeros(3))

    if 'ext_force_base' not in arm.output_mappings:
        outputs['ext_force_base'] = ir.utils.const_property(np.zeros(3))

    return _state_mapping(ir.compose(*components, inputs=inputs, outputs=outputs))


@ir.config(camera=positronic.cfg.hardware.camera.merged)
def umi(camera: Optional[ir.ControlSystem] = None):
    from positronic.drivers.umi import UmiCS  # noqa: CO415
    umi = UmiCS()
    components = [umi]
    outputs = {'frame': ir.OutputPort.Stub()}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    inputs = {'target_position': (umi, 'tracker_position'), 'target_grip': (umi, 'target_grip'), 'reset': None}

    outputs.update({
        'position': umi.outs.ee_position,
        'grip': umi.outs.grip,
        'status': ir.OutputPort.Stub(),
        'ext_force_ee': ir.utils.const_property(np.zeros(3)),
        'ext_force_base': ir.utils.const_property(np.zeros(3)),
        'metadata': umi.outs.metadata
    })

    res = ir.compose(*components, inputs=inputs, outputs=outputs)
    res = _state_mapping(res)
    return res


@ir.config(
    roboarm=positronic.cfg.hardware.roboarms.franka_ik,
    camera=positronic.cfg.hardware.camera.merged,
)
def franka(
    roboarm: ir.ControlSystem,
    camera: ir.ControlSystem | None = None,
):
    components = [roboarm]
    outputs = {'frame': ir.OutputPort.Stub()}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    async def metadata():
        return ir.Message(data={'source': 'franka'})

    inputs = {
        'target_position': (roboarm, 'target_position'),
        'target_grip': (roboarm, 'target_grip'),
        'reset': (roboarm, 'reset')
    }

    outputs.update({
        'robot_position': roboarm.outs.position,
        'joint_positions': roboarm.outs.joint_positions,
        'grip': roboarm.outs.grip,
        'robot_status': ir.OutputPort.Stub(),
        'ext_force_ee': roboarm.outs.ext_force_ee,
        'ext_force_base': roboarm.outs.ext_force_base,
        'metadata': metadata,
    })

    res = ir.compose(*components, inputs=inputs, outputs=outputs)
    res = _state_mapping(res)
    return res
