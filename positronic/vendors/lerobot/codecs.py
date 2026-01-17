"""LeRobot codecs (observation encoder + action decoder pairs)."""

from positronic.cfg import codecs

# LeRobot codec variants using base codec config
eepose_absolute = codecs.codec.override(observation=codecs.eepose, action=codecs.absolute_position(rotation_rep=None))

joints_absolute = codecs.codec.override(observation=codecs.joints, action=codecs.absolute_position(rotation_rep=None))
