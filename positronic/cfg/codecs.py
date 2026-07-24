"""Configuration for policy codecs (observation encoders and action decoders)."""

import configuronic as cfn

from positronic import geom
from positronic.policy.codec import ChangeEEFrame
from positronic.policy.observation import ObservationCodec

RotRep = geom.Rotation.Representation

# The border codec that converts poses into a policy's own EE frame (e.g. DROID's ``droid_eef``). Applied
# client-side at serving via ``remote(codec=change_ee_frame.override(to=...))``; the ``compose`` ``ee_frame``
# param below folds it into the training pipeline so the recorded data is encoded in the same frame.
change_ee_frame = cfn.Config(ChangeEEFrame)


@cfn.config()
def general_obs(
    state_name: str, state_features: dict[str, int], image_mappings: dict[str, str], image_size: tuple[int, int]
):
    """General observation encoder for non-GR00T policies (OpenPI, ACT, etc.)."""
    state_dict = {state_name: state_features}
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    return ObservationCodec(state=state_dict, images=images)


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


@cfn.config(fps=15.0, horizon=None, binarize_grip=None, flip_grip=False, ee_frame=None)
def compose(
    obs,
    action,
    fps: float,
    horizon: float | None,
    binarize_grip: tuple[str, ...] | None,
    flip_grip: bool,
    ee_frame: str | None,
):
    """Compose observation and action codecs with timing and optional grip binarization.

    ``flip_grip`` serves checkpoints that speak the inverted grip convention (see ``FlipGrip``). ``ee_frame``
    converts poses into the policy's own EE frame (see ``ChangeEEFrame``); folded in for training so the recorded
    data is encoded in that frame. At serving the frame conversion runs client-side instead (``change_ee_frame``),
    so leave ``ee_frame`` unset on the server's frame-agnostic codec.

    Layout::

        [ActionHorizon] | ActionTimestamp | [BinarizeGrip*] | [FlipGrip] | [ChangeEEFrame] | obs & action
    """
    from positronic.policy.codec import (
        ActionHorizon,
        ActionTimestamp,
        BinarizeGripInference,
        BinarizeGripTraining,
        FlipGrip,
    )

    result = obs & action
    if ee_frame is not None:
        result = ChangeEEFrame(to=ee_frame) | result
    if flip_grip:
        result = FlipGrip() | result
    if binarize_grip:
        result = BinarizeGripTraining(binarize_grip) | BinarizeGripInference() | result
    result = ActionTimestamp(fps=fps) | result
    if horizon is not None:
        result = ActionHorizon(horizon) | result
    return result


@cfn.config(rotation_rep=None, tgt_ee_pose_key='robot_command.pose', tgt_grip_key='target_grip')
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


@cfn.config(
    solver='dls_limits',
    tgt_ee_pose_key='robot_command.pose',
    tgt_grip_key='target_grip',
    current_q_key='robot_state.q',
    num_joints=7,
)
def ik_joints_action(solver, tgt_ee_pose_key, tgt_grip_key, current_q_key, num_joints):
    """Joint-space action codec that reconstructs target joints from EE targets via IK."""
    from positronic.drivers.roboarm.ik import DLSIKSolver, DLSIKSolverWithLimits, LMIKSolver
    from positronic.policy.action import AbsoluteJointsAction, IKJointsAction

    tgt_joints_key = 'robot_command.joints'
    solver_map = {'lm': LMIKSolver, 'dls': DLSIKSolver, 'dls_limits': DLSIKSolverWithLimits}
    ik = IKJointsAction(
        solver_cls=solver_map[solver],
        tgt_ee_pose_key=tgt_ee_pose_key,
        current_q_key=current_q_key,
        tgt_joints_key=tgt_joints_key,
    )
    return ik | AbsoluteJointsAction(tgt_joints_key=tgt_joints_key, tgt_grip_key=tgt_grip_key, num_joints=num_joints)
