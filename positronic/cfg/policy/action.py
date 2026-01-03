import configuronic as cfn

from positronic import geom

RotRep = geom.Rotation.Representation


@cfn.config(rotation_rep=None, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def absolute_position(rotation_rep: str | None, tgt_ee_pose_key: str, tgt_grip_key: str):
    """Unified GR00T action decoder for training and inference.

    Args:
        rotation_rep: Target rotation representation (e.g., 'rot6d'). None keeps QUAT (7D).
    """
    from positronic.policy.action import GrootActionDecoder

    rot_rep = RotRep(rotation_rep) if rotation_rep else None
    ee_dim = (rot_rep.shape if rot_rep else 4) + 3  # 3 for xyz translation

    result = GrootActionDecoder(rotation_rep=rot_rep, tgt_ee_pose_key=tgt_ee_pose_key, tgt_grip_key=tgt_grip_key)
    result.meta['gr00t_modality'] = {
        'action': {'ee_pose': {'start': 0, 'end': ee_dim}, 'grip': {'start': ee_dim, 'end': ee_dim + 1}}
    }
    result.meta['lerobot_features'] = {'action': {'shape': (ee_dim + 1,), 'names': ['actions'], 'dtype': 'float32'}}
    return result


groot = absolute_position.copy()
groot_rot6d = groot.override(rotation_rep='rot6d')

# TODO: We currently don't support absolute joint control, as collected datasets use cartesian control
# Two potential solutions:
# * Have a transform that computes IK (cartesian -> joint)
# * As most controllers do IK themselves, log target joints in the data collection


@cfn.config(num_joints=7)
def joint_delta(num_joints: int):
    from positronic.policy.action import JointDeltaAction

    result = JointDeltaAction(num_joints=num_joints)
    result.meta['lerobot_features'] = {'action': {'shape': (num_joints + 1,), 'names': ['actions'], 'dtype': 'float32'}}

    return result
