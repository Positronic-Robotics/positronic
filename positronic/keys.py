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
