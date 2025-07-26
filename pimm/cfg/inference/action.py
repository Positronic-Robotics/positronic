import configuronic as cfn
import geom
from pimm.inference.action import RelativeRobotPositionAction, UMIRelativeRobotPositionAction


umi_relative = cfn.Config(
    UMIRelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)

relative_robot_position = cfn.Config(
    RelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)
