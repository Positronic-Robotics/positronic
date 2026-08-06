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

# The arm's state channel, and the signals its serializer unfolds it into.
ROBOT_STATE = 'robot_state'
JOINTS = f'{ROBOT_STATE}.q'
JOINT_VEL = f'{ROBOT_STATE}.dq'
EE_POSE = f'{ROBOT_STATE}.ee_pose'
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

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'
