import configuronic as cfn
from positronic import geom


@cfn.config(rotation_representation=geom.Rotation.Representation.ROTVEC, offset=1)
def relative_robot_position(rotation_representation: geom.Rotation.Representation, offset: int):
    from positronic.policy.action import RelativeRobotPositionAction
    return RelativeRobotPositionAction(offset=offset, rotation_representation=rotation_representation)


@cfn.config(rotation_representation=geom.Rotation.Representation.QUAT)
def absolute_position(rotation_representation: geom.Rotation.Representation):
    from positronic.policy.action import AbsolutePositionAction
    return AbsolutePositionAction(rotation_representation=rotation_representation)


@cfn.config()
def absolute_joint_position():
    from positronic.policy.action import AbsoluteJointPositionAction
    return AbsoluteJointPositionAction()
