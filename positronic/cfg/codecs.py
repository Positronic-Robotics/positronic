"""Configuration for policies codecs (observation encoders and action decoders)."""

import configuronic as cfn

from positronic import geom
from positronic.policy.observation import SimpleObservationEncoder


@cfn.config()
def codec(observation, action):
    """Base codec config that pairs an observation encoder with an action decoder."""
    from positronic.policy import Codec

    return Codec(observation=observation, action=action)


@cfn.config()
def general(
    state_name: str,
    state_features: dict[str, int],
    image_mappings: dict[str, str],
    image_size: tuple[int, int],
    task_field: str | None = 'task',
):
    """General observation encoder for non-GR00T policies (OpenPI, ACT, etc.)."""
    state_dict = {state_name: state_features}
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    result = SimpleObservationEncoder(state=state_dict, images=images, task_field=task_field)

    state_dim = sum(state_features.values())
    lerobot_features = {state_name: {'shape': (state_dim,), 'names': list(state_features.keys()), 'dtype': 'float32'}}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features

    return result


eepose_grip = general.override(
    state_name='observation.state', state_features={'robot_state.ee_pose': 7, 'grip': 1}, image_size=(224, 224)
)

joints_grip = general.override(
    state_name='observation.state', state_features={'robot_state.q': 7, 'grip': 1}, image_size=(224, 224)
)

eepose_grip_joints = general.override(
    state_name='observation.state',
    state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7},
    image_size=(224, 224),
)

eepose = eepose_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
joints = joints_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
eepose_q = eepose_grip_joints.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)


RotRep = geom.Rotation.Representation


@cfn.config(rotation_rep=None, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip', action_horizon=None)
def absolute_position(rotation_rep: str | None, tgt_ee_pose_key: str, tgt_grip_key: str, action_horizon: int | None):
    """Absolute position action decoder for ACT/OpenPI.

    Decodes from {'action': vector} format.
    """
    from positronic.policy.action import AbsolutePositionAction

    rot_rep = RotRep(rotation_rep) if rotation_rep else RotRep.QUAT
    ee_dim = rot_rep.size + 3

    result = AbsolutePositionAction(
        tgt_ee_pose_key, tgt_grip_key, rotation_representation=rot_rep, action_horizon=action_horizon
    )
    result.meta['lerobot_features'] = {'action': {'shape': (ee_dim + 1,), 'names': ['actions'], 'dtype': 'float32'}}
    return result


# TODO: We currently don't support absolute joint control, as collected datasets use cartesian control
# Two potential solutions:
# * Have a transform that computes IK (cartesian -> joint)
# * As most controllers do IK themselves, log target joints in the data collection


@cfn.config(num_joints=7, action_horizon=None)
def joint_delta(num_joints: int, action_horizon: int | None):
    from positronic.policy.action import JointDeltaAction

    result = JointDeltaAction(num_joints=num_joints, action_horizon=action_horizon)
    result.meta['lerobot_features'] = {'action': {'shape': (num_joints + 1,), 'names': ['actions'], 'dtype': 'float32'}}

    return result
