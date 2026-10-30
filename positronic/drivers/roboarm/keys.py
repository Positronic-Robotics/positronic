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

# The gravity torque factor per joint that an arm's controller reports, keyed in an episode by its joint signal.
GRAVITY_COMP_FACTOR = 'gravity_comp_factor'
