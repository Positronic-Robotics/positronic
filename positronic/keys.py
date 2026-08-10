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


def is_robot_command(name: str) -> bool:
    """Whether ``name`` is in the robot-command family: ``robot_command``, or an arm's ``robot_command.{side}``.

    ``TARGET_EE_POSE`` and ``TARGET_JOINTS`` are in the family by name while carrying a vector.
    """
    return name == ROBOT_COMMAND or name.startswith(f'{ROBOT_COMMAND}.')


JOINTS = 'robot_state.q'
JOINT_VEL = 'robot_state.dq'
EE_POSE = 'robot_state.ee_pose'
GRIP = 'grip'
TASK = 'task'
WRIST_IMAGE = 'image.wrist'
EXTERIOR_IMAGE = 'image.exterior'

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
# is the env's judgement, carried by every trial an env can judge — either from the terminal payload its
# adapter returns, or as False when the budget ran out before one came. Absence means no env judged it.
EVAL_SUCCESS = 'eval.success'
EVAL_TERMINATED = 'eval.terminated'
