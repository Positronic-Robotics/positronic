# Configuration for the robotics environment
from typing import Optional

import numpy as np

from cfg import store, builds
import cfg.hardware.camera
import ironic as ir

cfg.hardware.camera.store.add_to_hydra_store()


def _physical_env(_roboarm: ir.ControlSystem, _camera: ir.ControlSystem):
    pass


physical_env = builds(_physical_env, populate_full_signature=True)


def _state_mapping(env: ir.ControlSystem):
    robot_properties = {}
    for name, mapping in env.output_mappings.items():
        if name == 'metadata':
            continue
        if ir.is_property(mapping):
            robot_properties[name] = mapping

    return ir.extend(env, {'state': ir.utils.properties_dict(**robot_properties)})


def _umi_env(camera: Optional[ir.ControlSystem] = None):
    from drivers.umi import UmiCS  # noqa: CO415
    umi = UmiCS()
    components = [umi]
    outputs = {'frame': ir.OutputPort.Stub()}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    inputs = {'target_position': (umi, 'tracker_position'), 'target_grip': (umi, 'target_grip'), 'reset': None}

    async def fake_ext_force_ee():
        return ir.Message(data=np.zeros(3))

    outputs.update({
        'robot_position': umi.outs.ee_position,
        'grip': umi.outs.grip,
        'robot_status': ir.OutputPort.Stub(),
        'ext_force_ee': fake_ext_force_ee,
        'metadata': umi.outs.metadata
    })

    res = ir.compose(*components, inputs=inputs, outputs=outputs)
    res = _state_mapping(res)
    return res


umi_env = builds(
    _umi_env,
    camera=cfg.hardware.camera.merged,
    populate_full_signature=True
)

store = store(group="env")
umi_env = store(umi_env(), name="umi")
