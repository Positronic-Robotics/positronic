import configuronic as cfgc
import geom
from positronic.inference.action import (RelativeRobotPositionAction,
                                         UMIRelativeRobotPositionAction)

umi_relative = cfgc.Config(
    UMIRelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)

relative_robot_position = cfgc.Config(
    RelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)
