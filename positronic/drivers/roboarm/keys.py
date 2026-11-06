"""The keys of the robot model an episode carries, of the frame each of its pose signals is stated in, and of the
settings each arm runs with."""

# The robot model, carried in episode statics for the transforms that solve against it (``IKJointsAction``).
URDF = 'urdf'
JOINT_NAMES = 'joint_names'
# The mesh files that ``URDF`` names, by file name, as bytes.
MESHES = 'meshes'
# The gripper spec the viewer drives: the signal it reads, the joints it moves, their travel at full closure.
GRIPPER = 'gripper'

# Where each pose signal sits, keyed by the signal, as dumped ``models.EEFrame`` entries. A rig, a recording and a
# policy that reads poses state it.
EE_FRAMES = 'ee_frames'

# The gravity torque factor per joint that an arm's controller reports.
GRAVITY_COMP_FACTOR = 'gravity_comp_factor'
# The mass of the end effector that an arm's controller compensates gravity with, in kg; None keeps its model's.
EE_MASS = 'ee_mass'
# The Coulomb friction torque per joint that an arm's controller adds, in Nm, and whether it adds it.
COULOMB_FRICTION = 'coulomb_friction'
USE_COULOMB_FRICTION = 'use_coulomb_friction'
# How an arm's controller fades that friction at low speed, as a dumped ``FrictionFade``: none below one speed and
# all of it from another, in rad/s. None adds the friction at any speed.
COULOMB_FRICTION_FADE = 'coulomb_friction_fade'
