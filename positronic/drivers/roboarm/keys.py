"""The keys of the robot model an episode carries, and of the frame each of its pose signals is stated in."""

# The robot model, carried in episode statics for the transforms that solve against it (``IKJointsAction``).
URDF = 'urdf'
JOINT_NAMES = 'joint_names'
# The gripper spec the viewer drives: the signal it reads, the joints it moves, their travel at full closure.
GRIPPER = 'gripper'

# Where each pose signal sits, keyed by the signal, as dumped ``models.EEFrame`` entries. A rig, a recording and a
# policy that reads poses state it; a policy that states none speaks joints only.
EE_FRAMES = 'ee_frames'
