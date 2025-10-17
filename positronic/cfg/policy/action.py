import configuronic as cfn

from positronic import geom

RotRep = geom.Rotation.Representation


@cfn.config(rotation_representation=RotRep.QUAT, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def absolute_position(rotation_representation: RotRep, tgt_ee_pose_key: str, tgt_grip_key: str):
    from positronic.policy.action import AbsolutePositionAction

    return AbsolutePositionAction(tgt_ee_pose_key, tgt_grip_key, rotation_representation=rotation_representation)


@cfn.config(num_joints=7)
def joint_delta(num_joints: int):
    from positronic.policy.action import JointDeltaAction

    return JointDeltaAction(num_joints=num_joints)
