import geom
import ironic as ir
from positronic.inference.action import UMIRelativeRobotPositionAction, RelativeRobotPositionAction


umi_relative = ir.Config(
    UMIRelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)

relative_robot_position = ir.Config(
    RelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)
