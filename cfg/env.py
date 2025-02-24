# Configuration for the robotics environment

from typing import Dict, Optional

from hydra_zen import builds, store
import numpy as np

import cfg.hardware.camera
import ironic as ir

def _physical_env(roboarm: ir.ControlSystem, camera: ir.ControlSystem):
    pass

physical_env = builds(_physical_env, populate_full_signature=True)

def _state_mapping(env: ir.ControlSystem, state_mappings: Dict[str, str]):
    robot_properties = {}
    for stored_name, property_name in state_mappings.items():
        robot_properties[stored_name] = getattr(env.outs, property_name)

    return ir.extend(env, {'state': ir.utils.properties_dict(**robot_properties)})


def _umi_env(camera: Optional[ir.ControlSystem] = None, state_mappings: Optional[Dict[str, str]] = None):
    from drivers.umi import UmiCS
    umi = UmiCS()
    components = [umi]
    outputs = {'frame': None}

    if camera is not None:
        components.append(camera)
        outputs['frame'] = camera.outs.frame

    inputs = {'target_position': (umi, 'tracker_position'),
              'target_grip': (umi, 'target_grip'),
              'reset': None}

    async def fake_ext_force_ee():
        return ir.Message(data=np.zeros(3))

    outputs.update({'robot_position': umi.outs.ee_position,
                    'grip': umi.outs.grip,
                    'robot_status': None,
                    'ext_force_ee': fake_ext_force_ee,
                    'metadata': umi.outs.metadata})

    res = ir.compose(*components, inputs=inputs, outputs=outputs)

    if state_mappings is not None:
        res = _state_mapping(res, state_mappings)
    return res


sm_store = store(group="env/state_mappings")
sm_store({
    'robot_position': 'robot_position',
    # 'robot_position_translation': 'robot_position_translation',
    # 'robot_position_quaternion': 'robot_position_quaternion'
}, name="umi")
sm_store({
    'robot_joints': 'joint_positions',
    'ext_force_ee': 'ext_force_ee',
    'ext_force_base': 'ext_force_base',
    'robot_position_translation': 'robot_position_translation',
    'robot_position_quaternion': 'robot_position_quaternion',
    'grip': 'grip'
}, name="franka")
sm_store.add_to_hydra_store()


umi_env = builds(_umi_env, populate_full_signature=True,
                hydra_defaults=["_self_",
                                {"/hardware/cameras@camera": "merged"},
                                {"state_mappings": "umi"}])

env_store = store(group="env")
env_store(umi_env(), name="umi")

env_store.add_to_hydra_store()
