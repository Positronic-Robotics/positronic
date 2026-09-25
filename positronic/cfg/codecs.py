"""Configuration for policy codecs (observation encoders and action decoders)."""

import configuronic as cfn

from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy import keys as policy_keys
from positronic.policy.codec import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    FlipGrip,
    Metadata,
    SetControlMode,
    StateFeature,
    pose_feature,
)
from positronic.policy.observation import ObservationCodec

RotRep = geom.Rotation.Representation


@cfn.config()
def general_obs(
    state_name: str,
    state_features: dict[str, StateFeature],
    image_mappings: dict[str, str],
    image_size: tuple[int, int],
):
    """General observation encoder for non-GR00T policies (OpenPI, ACT, etc.)."""
    state_dict = {state_name: state_features}
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    return ObservationCodec(state=state_dict, images=images)


eepose_grip_obs = general_obs.override(
    state_name='observation.state', state_features={keys.EE_POSE: pose_feature(), keys.GRIP: 1}, image_size=(224, 224)
)

joints_grip_obs = general_obs.override(
    state_name='observation.state', state_features={keys.JOINTS: 7, keys.GRIP: 1}, image_size=(224, 224)
)

eepose_grip_joints_obs = general_obs.override(
    state_name='observation.state',
    state_features={keys.EE_POSE: pose_feature(), keys.GRIP: 1, keys.JOINTS: 7},
    image_size=(224, 224),
)

eepose_obs = eepose_grip_obs.override(
    image_mappings={'observation.images.left': keys.WRIST_IMAGE, 'observation.images.side': keys.EXTERIOR_IMAGE}
)
joints_obs = joints_grip_obs.override(
    image_mappings={'observation.images.left': keys.WRIST_IMAGE, 'observation.images.side': keys.EXTERIOR_IMAGE}
)
eepose_joints_obs = eepose_grip_joints_obs.override(
    image_mappings={'observation.images.left': keys.WRIST_IMAGE, 'observation.images.side': keys.EXTERIOR_IMAGE}
)


@cfn.config(binarize_grip=None, flip_grip=False, ee_frame=None)
def compose_data(
    obs, action, binarize_grip: tuple[str, ...] | None, flip_grip: bool, ee_frame: geom.Transform3D | None
):
    """Compose observation and action conversions, with optional grip and frame conversions.

    ``flip_grip`` serves checkpoints that speak the inverted grip convention (see ``FlipGrip``). ``ee_frame``
    places the end-effector frame the checkpoint speaks relative to ``DEFAULT_FRAME`` (``models.DROID_EE_FRAME``)
    and re-expresses a dataset in it for training (see ``ChangeEEFrame``); serving declares the conversion in
    the pipeline instead, so leave it unset there.

    """
    result = obs & action
    if ee_frame is not None:
        result = ChangeEEFrame(ee_frame) | result
    if flip_grip:
        result = FlipGrip() | result
    if binarize_grip:
        result = BinarizeGripTraining(binarize_grip) | BinarizeGripInference() | result
    return result


@cfn.config(training_fps=15.0, binarize_grip=None, flip_grip=False, ee_frame=None)
def compose(
    obs,
    action,
    training_fps: float,
    binarize_grip: tuple[str, ...] | None,
    flip_grip: bool,
    ee_frame: geom.Transform3D | None,
):
    """Data conversions with the sampling cadence recorded in training metadata."""
    result = compose_data(obs=obs, action=action, binarize_grip=binarize_grip, flip_grip=flip_grip, ee_frame=ee_frame)
    return Metadata({policy_keys.ACTION_FPS: training_fps}) | result


@cfn.config(rotation_rep=None, tgt_ee_pose_key=keys.TARGET_EE_POSE, tgt_grip_key=keys.TARGET_GRIP)
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


@cfn.config()
def droid_execution(action):
    """Wrap an action codec so its chunks execute under impedance, as DROID-pretrained checkpoints expect."""
    return SetControlMode(DROID_IMPEDANCE) | action


@cfn.config()
def phail_v1_execution(action):
    """Wrap an action codec so its chunks execute under the position control PhAIL v1 was trained on."""
    return SetControlMode(roboarm_command.PositionControl()) | action


traj_ee_action = absolute_pos_action.override(tgt_ee_pose_key=keys.EE_POSE, tgt_grip_key=keys.GRIP)


@cfn.config(
    solver='dls_limits',
    tgt_ee_pose_key=keys.TARGET_EE_POSE,
    tgt_grip_key=keys.TARGET_GRIP,
    current_q_key=keys.JOINTS,
    num_joints=7,
)
def ik_joints_action(solver, tgt_ee_pose_key, tgt_grip_key, current_q_key, num_joints):
    """Joint-space action codec that reconstructs target joints from EE targets via IK."""
    from positronic.drivers.roboarm.ik import DLSIKSolver, DLSIKSolverWithLimits, LMIKSolver
    from positronic.policy.action import AbsoluteJointsAction, IKJointsAction

    tgt_joints_key = keys.TARGET_JOINTS
    solver_map = {'lm': LMIKSolver, 'dls': DLSIKSolver, 'dls_limits': DLSIKSolverWithLimits}
    ik = IKJointsAction(
        solver_cls=solver_map[solver],
        tgt_ee_pose_key=tgt_ee_pose_key,
        current_q_key=current_q_key,
        tgt_joints_key=tgt_joints_key,
    )
    return ik | AbsoluteJointsAction(tgt_joints_key=tgt_joints_key, tgt_grip_key=tgt_grip_key, num_joints=num_joints)
