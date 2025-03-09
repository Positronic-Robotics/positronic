import geom
import ironic as ir
from positronic.inference.action import UMIRelativeRobotPositionAction


umi_relative = ir.Config(
    UMIRelativeRobotPositionAction,
    rotation_representation=geom.Rotation.Representation.ROTVEC,
    offset=1
)