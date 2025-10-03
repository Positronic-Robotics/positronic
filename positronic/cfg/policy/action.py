import configuronic as cfn

from positronic import geom

RotRep = geom.Rotation.Representation


@cfn.config(rotation_representation=RotRep.QUAT, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def absolute_position(rotation_representation: RotRep, tgt_ee_pose_key: str, tgt_grip_key: str):
    from positronic.policy.action import AbsolutePositionAction

    return AbsolutePositionAction(
        tgt_ee_pose_key,
        tgt_grip_key,
        rotation_representation=rotation_representation,
    )


@cfn.config()
def absolute_joint_position():
    from positronic.policy.action import AbsoluteJointPositionAction

    return AbsoluteJointPositionAction()
