"""The keys of the robot model an episode carries, and of the frame its poses are stated in."""

# The robot model, carried in episode statics for the transforms that solve against it (``IKJointsAction``).
# ``CONTROL_FRAME`` names the frame in ``URDF`` that the embodiment reports ``EE_POSE`` in; every embodiment
# declares it as ``DEFAULT_FRAME``, and datasets recorded before that convention name their own frame.
URDF = 'urdf'
CONTROL_FRAME = 'control_frame'
JOINT_NAMES = 'joint_names'
# The gripper spec the viewer drives: the signal it reads, the joints it moves, their travel at full closure.
GRIPPER = 'gripper'

# How many times the arm's driver saw a safe input on the control box go from clear to triggered, up to the
# end of this episode. A triggered safe input prohibits motion. A count that rose across an episode therefore
# says the rig stopped the arm, which is what tells an episode a person ended because the attempt failed apart
# from one they ended because the arm had stopped moving.
SAFE_STOP_TRIPS = 'robot.safe_stop_trips'

# Where the episode's poses sit relative to ``DEFAULT_FRAME``, as a ``[tx,ty,tz,qw,qx,qy,qz]`` transform.
# Absent means they are in that frame itself; ``ChangeEEFrame`` writes it when it moves them.
EE_FRAME = 'ee_frame'
