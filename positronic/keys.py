"""Canonical names on the positronic embodiment/inference wire.

Every sim adapter and embodiment produces them and every vendor codec consumes them. They are defined
here once, in a leaf module with no positronic imports, so a rename is a single-site change the type
checker propagates instead of a string literal duplicated across codecs, configs, adapters and datasets.
A key one package owns lives in that package instead: a trial's in ``positronic.eval``, the robot
model's in ``drivers.roboarm.models``, the inference handshake's in ``policy.base``, a benchmark's in
its adapter.
"""

# The arm's command channel, and the signals a recorded command unfolds into. The suffixes are the
# serializer's (see ``Serializers.robot_command`` and ``expand_suffixed``), so the names derive from the
# channel rather than restating it.
ROBOT_COMMAND = 'robot_command'
TARGET_EE_POSE = f'{ROBOT_COMMAND}.pose'
TARGET_JOINTS = f'{ROBOT_COMMAND}.joints'

# The gripper's command channel: a scalar target beside the arm's ``ROBOT_COMMAND``.
TARGET_GRIP = 'target_grip'


def is_robot_command(name: str) -> bool:
    """Whether ``name`` is in the robot-command family: ``robot_command``, or an arm's ``robot_command.{side}``.

    ``TARGET_EE_POSE`` and ``TARGET_JOINTS`` are in the family by name while carrying a vector.
    """
    return name == ROBOT_COMMAND or name.startswith(f'{ROBOT_COMMAND}.')


# The arm's state channel, and the signals a recorded state unfolds into. As on the command side, the
# suffixes are ``Serializers.robot_state``'s, so the names derive from the channel rather than restating it.
ROBOT_STATE = 'robot_state'
JOINTS = f'{ROBOT_STATE}.q'
JOINT_VEL = f'{ROBOT_STATE}.dq'
EE_POSE = f'{ROBOT_STATE}.ee_pose'
# The arm's ``RobotStatus``. The suffix is named on its own because a consumer picks the entry out by it on a
# rig whose arms are ``robot_state.{side}``.
STATUS_SUFFIX = '.status'
ROBOT_STATUS = f'{ROBOT_STATE}{STATUS_SUFFIX}'
GRIP = 'grip'
TASK = 'task'
# The embodiment an observation came from, so a multi-embodiment policy can tell which robot it is driving.
DESCRIPTOR = 'descriptor'
# The prefix that identifies a camera on the wire: an embodiment declares its cameras by naming
# them this way, and every consumer picks them out of the observations by it.
IMAGE_PREFIX = 'image.'
WRIST_IMAGE = f'{IMAGE_PREFIX}wrist'
EXTERIOR_IMAGE = f'{IMAGE_PREFIX}exterior'

# The harness stamps each observation with the control clock's time (``OBS_TIME_NS``) and the wall
# clock's (``WALL_TIME_NS``); recording timelines and action scheduling read time back off them.
# ``ACTION_TIMESTAMP`` is where a decoded action carries its schedule slot, in seconds from the
# observation it answers.
OBS_TIME_NS = 'obs_time_ns'
WALL_TIME_NS = 'wall_time_ns'
ACTION_TIMESTAMP = 'timestamp'
