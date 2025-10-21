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
    result.meta['lerobot_features'] = {
        'action': {
            'shape': (8,),
            'names': [
                'rotation_0',
                'rotation_1',
                'rotation_2',
                'rotation_3',
                'translation_x',
                'translation_y',
                'translation_z',
                'grip',
            ],
            'dtype': 'float32',
        }
    }
    return result


@cfn.config(num_joints=7)
def joint_delta(num_joints: int):
    from positronic.policy.action import JointDeltaAction

    result = JointDeltaAction(num_joints=num_joints)
    result.meta['lerobot_features'] = {'action': {'shape': (num_joints + 1,), 'names': ['actions'], 'dtype': 'float32'}}

    return result
