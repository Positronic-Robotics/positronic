import configuronic as cfn

from positronic import geom

RotRep = geom.Rotation.Representation


@cfn.config(rotation_representation=RotRep.QUAT, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def absolute_position(rotation_representation: RotRep, tgt_ee_pose_key: str, tgt_grip_key: str):
    from positronic.policy.action import AbsolutePositionAction

    result = AbsolutePositionAction(tgt_ee_pose_key, tgt_grip_key, rotation_representation=rotation_representation)
    result.meta['gr00t_modality'] = {
        'action': {
            'target_robot_position_quaternion': {'start': 0, 'end': 4, 'rotation_type': 'quaternion'},
            'target_robot_position_translation': {'start': 4, 'end': 7},
            'target_grip': {'start': 7, 'end': 8},
        }
    }
    result.meta['lerobot_features'] = {'action': {'shape': (8,), 'names': ['actions'], 'dtype': 'float32'}}
    return result


# TODO: This currently does not work, as collected datasets use cartesian control
# Two potential solutions:
# * Have a trasnform that computes IK (cartesian -> joint)
# * As most controllers do IK themselves, log target joints in the data collection
@cfn.config(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip', num_joints=7)
def absolute_joints(tgt_joints_key: str, tgt_grip_key: str, num_joints: int):
    from positronic.policy.action import AbsoluteJointsAction

    result = AbsoluteJointsAction(tgt_joints_key, tgt_grip_key, num_joints=num_joints)
    result.meta['gr00t_modality'] = {
        'action': {
            'target_joint_positions': {'start': 0, 'end': num_joints},
            'target_grip': {'start': num_joints, 'end': num_joints + 1},
        }
    }
    result.meta['lerobot_features'] = {'action': {'shape': (num_joints + 1,), 'names': ['actions'], 'dtype': 'float32'}}
    return result


@cfn.config(num_joints=7)
def joint_delta(num_joints: int):
    from positronic.policy.action import JointDeltaAction

    result = JointDeltaAction(num_joints=num_joints)
    result.meta['lerobot_features'] = {'action': {'shape': (num_joints + 1,), 'names': ['actions'], 'dtype': 'float32'}}

    return result


@cfn.config()
def gr00t_oxe_droid():
    from positronic.policy.action import RelativeTargetPositionAction

    result = RelativeTargetPositionAction(rotation_representation=RotRep.ROTVEC)
    result.meta['gr00t_modality'] = {
        'action': {
            'eef_rotation_delta': {'start': 0, 'end': 3, 'rotation_type': 'axis_angle'},
            'eef_position_delta': {'start': 3, 'end': 6},
            'gripper_position': {'start': 6, 'end': 7},
        }
    }
    result.meta['lerobot_features'] = {'action': {'shape': (7,), 'names': ['actions'], 'dtype': 'float32'}}
    return result
