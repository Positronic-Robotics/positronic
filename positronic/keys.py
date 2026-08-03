"""Canonical raw observation-signal keys of the positronic embodiment/inference wire.

Every sim adapter and embodiment produces these keys and every vendor codec consumes them. They
are defined here once, in a leaf module with no positronic imports, so a rename is a single-site
change the type checker propagates instead of a string literal duplicated across codecs, evals,
configs, adapters and datasets.
"""

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
