import numpy as np

import geom
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.roboarms
import positronic.cfg.hardware.gripper
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
    from positronic.drivers.gripper.dh import DHGripper
    gripper = DHGripper(port='/dev/ttyUSB0')
    components = [arm, gripper]

    inputs = {
        'target_position': (arm, 'target_position'),
        'reset': (arm, 'reset'),
        'target_grip': (gripper, 'target_grip'),
    }

    outputs = {**arm.output_mappings, 'frame': ir.OutputPort.Stub(), 'grip': gripper.outs.grip}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    if 'ext_force_ee' not in arm.output_mappings:
        outputs['ext_force_ee'] = ir.utils.const_property(np.zeros(3))

    if 'ext_force_base' not in arm.output_mappings:
        outputs['ext_force_base'] = ir.utils.const_property(np.zeros(3))

    return _state_mapping(ir.compose(*components, inputs=inputs, outputs=outputs))


GRIPPER_REGISTRATION = geom.Transform3D(
    rotation=geom.Rotation.from_quat([0.88092353, 0.45079771, 0.11161464, -0.09108971]),
)


@ir.config(
    camera=positronic.cfg.hardware.camera.merged,
    registration_transform=GRIPPER_REGISTRATION,
    gripper=None,
)
def umi(camera: ir.ControlSystem, registration_transform: geom.Transform3D, gripper: ir.ControlSystem | None):
    from positronic.drivers.umi import UmiCS
    umi = UmiCS(registration_transform)
    components = [umi, camera]

    inputs = {'target_position': (umi, 'tracker_position'), 'target_grip': None}
    if gripper is not None:
        components.append(gripper)
        inputs['target_grip'] = (gripper, 'target_grip')

    outputs = {
        'frame': camera.outs.frame,
        'status': ir.OutputPort.Stub(),
        'metadata': umi.outs.metadata,
        'umi_left': umi.outs.umi_left,
        'umi_right': umi.outs.umi_right,
        'grip': gripper.outs.grip if gripper is not None else ir.OutputPort.Stub(),
    }

    res = ir.compose(*components, inputs=inputs, outputs=outputs)
    res = _state_mapping(res)
    return res


umi_gripper = umi.override(gripper=positronic.cfg.hardware.gripper.dh)


@ir.config(
    roboarm=positronic.cfg.hardware.roboarms.franka_ik,
    camera=positronic.cfg.hardware.camera.merged,
)
def franka(
    roboarm: ir.ControlSystem,
    camera: ir.ControlSystem | None = None,
):
    # TODO: we should get rid of this in favor of roboarm env
    from positronic.drivers.gripper.dh import DHGripper

    components = [roboarm]
    outputs = {'frame': ir.OutputPort.Stub()}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    async def metadata():
        return ir.Message(data={'source': 'franka'})

    gripper = DHGripper(port="/dev/ttyUSB0")
    components.append(gripper)

    inputs = {
        'target_position': (roboarm, 'target_position'),
        'target_grip': (gripper, 'target_grip'),
        'reset': (roboarm, 'reset')
    }

    outputs.update({
        'position': roboarm.outs.position,
        'joint_positions': roboarm.outs.joint_positions,
        'grip': gripper.outs.grip,
        'status': ir.OutputPort.Stub(),
        'ext_force_ee': roboarm.outs.ext_force_ee,
        'ext_force_base': roboarm.outs.ext_force_base,
        'metadata': metadata,
    })

    res = ir.compose(*components, inputs=inputs, outputs=outputs)
    res = _state_mapping(res)
    return res
