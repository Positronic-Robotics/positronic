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

# The gripper's command channel, alongside ``ROBOT_COMMAND``: an action carries it, an embodiment binds it,
# and a recorded episode holds it under this name.
TARGET_GRIP = 'target_grip'

# When an action falls due, in seconds. Every action an inference hands back carries it: the scheduling
# wrapper anchors it to the live clock and the harness converts it to the nanoseconds its command channels
# take. It names a field of the action dict rather than a signal, so it is a different contract from the
# ``timestamp`` column a stored vector signal is written under.
ACTION_TIMESTAMP = 'timestamp'

JOINTS = 'robot_state.q'
JOINT_VEL = 'robot_state.dq'
EE_POSE = 'robot_state.ee_pose'
GRIP = 'grip'
TASK = 'task'
WRIST_IMAGE = 'image.wrist'
EXTERIOR_IMAGE = 'image.exterior'
# The scene camera a sim binds beside the two an arm carries: a fixed view of the whole workspace.
AGENT_VIEW_IMAGE = 'image.agent_view'

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
