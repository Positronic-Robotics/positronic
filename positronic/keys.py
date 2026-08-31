"""Canonical names on the positronic embodiment/inference wire.

Every sim adapter and embodiment produces the signal keys and every vendor codec consumes them;
every policy reports the handshake fields and the analysis configs read them back off an episode. They
are defined here once, in a leaf module with no positronic imports, so a rename is a single-site
change the type checker propagates instead of a string literal duplicated across codecs, evals,
configs, adapters and datasets.
"""

# The arm's command channel, and the signals a recorded command unfolds into. The suffixes are the
# serializer's (see ``Serializers.robot_command`` and ``expand_suffixed``), so the names derive from the
# channel rather than restating it.
ROBOT_COMMAND = 'robot_command'
TARGET_EE_POSE = f'{ROBOT_COMMAND}.pose'
TARGET_JOINTS = f'{ROBOT_COMMAND}.joints'

# The gripper's command channel: a scalar target beside the arm's ``ROBOT_COMMAND``.
TARGET_GRIP = 'target_grip'

# The names of what a trial readies before it opens. ``Embodiment.prepare_handlers`` is keyed by them, and so
# is what a ``Task`` asks for. A rig with two arms names its arms ``arm.{side}``. ``SCENE`` means the world
# this trial runs in is ready, drawn by whichever handler the embodiment binds.
ARM = 'arm'
GRIPPER = 'gripper'
SCENE = 'scene'


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

# The robot model, carried in episode statics for the transforms that solve against it (``IKJointsAction``).
# ``CONTROL_FRAME`` names the frame in ``URDF`` that the embodiment reports ``EE_POSE`` in; every embodiment
# declares it as ``models.DEFAULT_FRAME``, and datasets recorded before that convention name their own frame.
URDF = 'urdf'
CONTROL_FRAME = 'control_frame'
JOINT_NAMES = 'joint_names'

# Where the episode's poses sit relative to ``models.DEFAULT_FRAME``, as a ``[tx,ty,tz,qw,qx,qy,qz]``
# transform. Absent means they are in that frame itself; ``ChangeEEFrame`` writes it when it moves them.
EE_FRAME = 'ee_frame'

# The 3D viewer's pointers into an episode: which signals carry joint angles, which carry poses. One arm per
# ``JOINT_SIGNALS`` entry, each running ``URDF`` and standing where ``MOUNTS`` places it, keyed by that signal.
JOINT_SIGNALS = 'joint_signals'
POSE_SIGNALS = 'pose_signals'
MOUNTS = 'mounts'

# The inference handshake: a policy reports these through its ``meta``, a remote policy nests the serving
# half under ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy
# at the top level and the vendor under ``SERVER``, so a reader composes a prefix with a field:
# f'{SERVER_META}.{TYPE}'.
TYPE = 'type'
CHECKPOINT_ID = 'checkpoint_id'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
HOST = 'host'
PORT = 'port'
SERVER = 'server'
# The wire half of the handshake: what the server declares for the rig to build and obey.
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'

# What a trial reports when it ends, in its episode's statics. The harness writes ``EVAL_TERMINATED``:
# True when a terminal was delivered inside the budget, False when the budget ran out. ``EVAL_SUCCESS``
# rides in the terminal payload an env's adapter returns, so an env that reports it only on success
# leaves it absent on failure — a reader defaults it rather than assuming a False.
EVAL_SUCCESS = 'eval.success'
EVAL_TERMINATED = 'eval.terminated'
# Whether the trial charged each model call the wall time it really took. True on a real rig whatever the
# task asked, since it cannot hold the world still while the model thinks.
EVAL_CHARGE_INFERENCE_TIME = 'eval.charge_inference_time'
# Who ended the trial, when it was not the task's own ground truth. An env's terminal leaves it absent.
EVAL_ENDED_BY = 'eval.ended_by'
ENDED_BY_OPERATOR = 'operator'

# The conditions the trial ran under, stamped into its episode's statics. ``EVAL_UNIVERSE`` is ``'sim'`` or
# ``'real'``; ``EVAL_TIMEOUT`` is absent from an episode whose task set no budget.
EVAL_UNIVERSE = 'eval.universe'
EVAL_EMBODIMENT = 'eval.embodiment'
EVAL_TIMEOUT = 'eval.timeout'
# The seconds the benchmark gives an episode of this task, carried in the trial's params. A config reads it
# to set the trial's deadline, which the harness records as ``EVAL_TIMEOUT``.
EVAL_EPISODE_LENGTH = 'eval.episode_length'

# A trial's place in its eval's sweep, stamped into every trial's params and recorded in its episode's
# statics. ``EVAL_SEED`` also rides an env's reset token, where absent means the env draws its own.
EVAL_SEED = 'eval.seed'
EVAL_TRIAL_INDEX = 'eval.trial_index'
EVAL_TRIAL_COUNT = 'eval.trial_count'

# The scene a trial runs. The benchmark's adapter names the task in the trial's params, from the task records
# the env answers with, and reads them back to build the env's reset token; the eval config adds the settings
# it owns. LIBERO names a task inside a suite plus the render and control settings its server caches an env
# by; RoboLab names a task and the phrasing of its instruction.
EVAL_SUITE = 'eval.suite'
EVAL_TASK_ID = 'eval.task_id'
EVAL_CAMERA_RESOLUTION = 'eval.camera_resolution'
EVAL_CONTROL_MODE = 'eval.control_mode'
EVAL_SETTLE_STEPS = 'eval.settle_steps'
EVAL_TASK = 'eval.task'
EVAL_INSTRUCTION_TYPE = 'eval.instruction_type'
