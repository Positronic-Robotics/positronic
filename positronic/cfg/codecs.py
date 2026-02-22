"""Configuration for policy codecs (observation encoders and action decoders)."""

import configuronic as cfn

from positronic import geom
from positronic.policy.observation import ObservationCodec

RotRep = geom.Rotation.Representation


@cfn.config()
def general_obs(
    state_name: str,
    state_features: dict[str, int],
    image_mappings: dict[str, str],
    image_size: tuple[int, int],
    task_field: str | None = 'task',
):
    """General observation encoder for non-GR00T policies (OpenPI, ACT, etc.)."""
    state_dict = {state_name: state_features}
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    return ObservationCodec(state=state_dict, images=images, task_field=task_field)


eepose_grip_obs = general_obs.override(
    state_name='observation.state', state_features={'robot_state.ee_pose': 7, 'grip': 1}, image_size=(224, 224)
)

joints_grip_obs = general_obs.override(
    state_name='observation.state', state_features={'robot_state.q': 7, 'grip': 1}, image_size=(224, 224)
)

eepose_grip_joints_obs = general_obs.override(
    state_name='observation.state',
    state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7},
    image_size=(224, 224),
)

eepose_obs = eepose_grip_obs.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
joints_obs = joints_grip_obs.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
eepose_joints_obs = eepose_grip_joints_obs.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)


@cfn.config(fps=15.0, horizon=1.0, binarize_grip=None)
def compose(obs, action, fps: float, horizon: float | None, binarize_grip: tuple[str, ...] | None):
    """Compose observation and action codecs with timing via ``|``."""
    from positronic.policy.codec import ActionTiming, BinarizeGripInference, BinarizeGripTraining

    result = obs | action
    if binarize_grip:
        result = BinarizeGripTraining(binarize_grip) | BinarizeGripInference() | result
    return ActionTiming(fps=fps, horizon_sec=horizon) | result


@cfn.config(rotation_rep=None, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def absolute_pos_action(rotation_rep: str | None, tgt_ee_pose_key: str, tgt_grip_key: str):
    """Absolute position action codec for ACT/OpenPI."""
    from positronic.policy.action import AbsolutePositionAction

    rot_rep = RotRep(rotation_rep) if rotation_rep else RotRep.QUAT
    return AbsolutePositionAction(tgt_ee_pose_key, tgt_grip_key, rotation_rep=rot_rep)


@cfn.config(num_joints=7)
def absolute_joints_action(tgt_joints_key: str, tgt_grip_key: str, num_joints: int):
    """Absolute joint position action codec."""
    from positronic.policy.action import AbsoluteJointsAction

    return AbsoluteJointsAction(tgt_joints_key, tgt_grip_key, num_joints=num_joints)


@cfn.config(num_joints=7)
def joint_delta_action(num_joints: int):
    from positronic.policy.action import JointDeltaAction

    return JointDeltaAction(num_joints=num_joints)


traj_ee_action = absolute_pos_action.override(tgt_ee_pose_key='robot_state.ee_pose', tgt_grip_key='grip')
