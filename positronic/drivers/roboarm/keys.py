"""The keys of the robot model an episode carries, and of the frame its poses are stated in."""

# The robot model, carried in episode statics for the transforms that solve against it (``IKJointsAction``).
# ``CONTROL_FRAME`` names the frame in ``URDF`` that the embodiment reports ``EE_POSE`` in; every embodiment
# declares it as ``DEFAULT_FRAME``, and datasets recorded before that convention name their own frame.
URDF = 'urdf'
CONTROL_FRAME = 'control_frame'
JOINT_NAMES = 'joint_names'

# Where the episode's poses sit relative to ``DEFAULT_FRAME``, as a ``[tx,ty,tz,qw,qx,qy,qz]`` transform.
# Absent means they are in that frame itself; ``ChangeEEFrame`` writes it when it moves them.
EE_FRAME = 'ee_frame'
