from enum import Enum


class RobotStatus(Enum):
    INVALID = "invalid"
    AVAILABLE = "available"
    RESETTING = "resetting"
