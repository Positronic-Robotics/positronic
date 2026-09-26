"""Canonical names on the positronic embodiment/inference wire.

Every sim adapter and embodiment produces them and every vendor codec consumes them. They are defined
here once, in a leaf module with no positronic imports, so a rename is a single-site change the type
checker propagates instead of a string literal duplicated across codecs, configs, adapters and datasets.
A key one package owns lives in that package's own ``keys`` module instead: a trial's in ``eval.keys``,
the robot model's in ``drivers.roboarm.keys``, and so on.
"""

JOINTS_SUFFIX = '.q'
JOINT_VEL_SUFFIX = '.dq'
EE_POSE_SUFFIX = '.ee_pose'
POSE_SUFFIX = '.pose'
STATUS_SUFFIX = '.status'

# The arm's command channel, and the signals a recorded command unfolds into. The suffixes are the
# serializer's (see ``Serializers.robot_command`` and ``expand_suffixed``), so the names derive from the
# channel rather than restating it.
ROBOT_COMMAND = 'robot_command'
TARGET_EE_POSE = f'{ROBOT_COMMAND}{POSE_SUFFIX}'
TARGET_JOINTS = f'{ROBOT_COMMAND}.joints'

# The gripper's command channel: a scalar target beside the arm's ``ROBOT_COMMAND``.
TARGET_GRIP = 'target_grip'


def is_robot_command(name: str) -> bool:
    """Whether ``name`` is in the robot-command family: ``robot_command``, or an arm's ``robot_command.{side}``.

    ``TARGET_EE_POSE`` and ``TARGET_JOINTS`` are in the family by name while carrying a vector.
    """
    return name == ROBOT_COMMAND or name.startswith(f'{ROBOT_COMMAND}.')


def arm_channel(channel: str, arm: str | None) -> str:
    """``channel`` for ``arm``: the bare channel when the arm is unnamed, ``channel.{arm}`` otherwise."""
    return channel if arm is None else f'{channel}.{arm}'


# The arms of a two-arm embodiment, left first.
LEFT_ARM, RIGHT_ARM = 'left', 'right'
BIMANUAL_ARMS = (LEFT_ARM, RIGHT_ARM)


# The arm's state channel, and the signals a recorded state unfolds into. As on the command side, the
# suffixes are ``Serializers.robot_state``'s, so the names derive from the channel rather than restating it.
ROBOT_STATE = 'robot_state'
JOINTS = f'{ROBOT_STATE}{JOINTS_SUFFIX}'
JOINT_VEL = f'{ROBOT_STATE}{JOINT_VEL_SUFFIX}'
EE_POSE = f'{ROBOT_STATE}{EE_POSE_SUFFIX}'
ROBOT_STATUS = f'{ROBOT_STATE}{STATUS_SUFFIX}'
GRIP = 'grip'
TASK = 'task'
# The embodiment an observation came from, so a multi-embodiment policy can tell which robot it is driving.
DESCRIPTOR = 'descriptor'
# The prefix that identifies a camera on the wire: an embodiment declares its cameras by naming
# them this way, and every consumer picks them out of the observations by it.
IMAGE_PREFIX = 'image.'
WRIST_IMAGE = f'{IMAGE_PREFIX}wrist'
WRIST_LEFT_IMAGE = f'{IMAGE_PREFIX}wrist_left'
WRIST_RIGHT_IMAGE = f'{IMAGE_PREFIX}wrist_right'
EXTERIOR_IMAGE = f'{IMAGE_PREFIX}exterior'
EXTERIOR_IMAGE_2 = f'{IMAGE_PREFIX}exterior_2'
