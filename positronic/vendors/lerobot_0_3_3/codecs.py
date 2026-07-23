"""LeRobot codecs (observation encoder | action decoder pairs)."""

from positronic import keys
from positronic.cfg import codecs

ee = codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action, horizon=1.0)
joints = ee.override(obs=codecs.joints_obs)

# Trajectory variants: use actual robot trajectory as action target instead of commanded targets
ee_traj = ee.override(action=codecs.traj_ee_action, binarize_grip=(keys.GRIP,))

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_traj = codecs.compose.override(
    obs=codecs.joints_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP),
    binarize_grip=(keys.GRIP,),
    horizon=1.0,
)

# IK variants: reconstruct joint targets from recorded EE targets via IK
joints_ik = ee.override(obs=codecs.joints_obs, action=codecs.ik_joints_action)
joints_ik_sim = joints_ik.override(**{'action.solver': 'lm'})
