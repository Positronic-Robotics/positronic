"""LeRobot codecs (observation encoder + action decoder pairs)."""

import configuronic as cfn

from positronic.cfg.policy import action, observation


@cfn.config()
def eepose_absolute():
    """LeRobot codec: EE pose + grip observation with absolute position actions."""
    return {'observation': observation.eepose, 'action': action.absolute_position(rotation_rep=None)}


@cfn.config()
def joints_absolute():
    """LeRobot codec: Joints + grip observation with absolute position actions."""
    return {'observation': observation.joints, 'action': action.absolute_position(rotation_rep=None)}
