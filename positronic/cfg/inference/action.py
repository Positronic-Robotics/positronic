import configuronic as cfn
from positronic import geom


@cfn.config(rotation_representation=geom.Rotation.Representation.ROTVEC, offset=1)
def umi_relative(rotation_representation: geom.Rotation.Representation, offset: int):
    from positronic.inference.action import UMIRelativeRobotPositionAction
    return UMIRelativeRobotPositionAction(offset=offset, rotation_representation=rotation_representation)


@cfn.config(rotation_representation=geom.Rotation.Representation.ROTVEC, offset=1)
def relative_robot_position(rotation_representation: geom.Rotation.Representation, offset: int):
    from positronic.inference.action import RelativeRobotPositionAction
    return RelativeRobotPositionAction(offset=offset, rotation_representation=rotation_representation)


@cfn.config(rotation_representation=geom.Rotation.Representation.QUAT)
def absolute_position(rotation_representation: geom.Rotation.Representation):
    from positronic.inference.action import AbsolutePositionAction
    return AbsolutePositionAction(rotation_representation=rotation_representation)