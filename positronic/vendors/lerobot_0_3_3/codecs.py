"""LeRobot codecs (observation encoder | action decoder pairs)."""

from positronic import keys
from positronic.cfg import codecs

ee = codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action, horizon=1.0)
phail_v1 = ee.override(action=codecs.phail_v1_execution.override(action=codecs.absolute_pos_action))
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

# The Trossen WidowX AI carries six arm joints where the codecs above assume seven, and a demonstration
# driven by a leader arm records the joints the leader asked for. Those are what the follower's own driver
# takes back, with no pose to solve for on the way in or out.
trossen_joints = codecs.compose.override(
    obs=codecs.joints_obs.override(state_features={keys.JOINTS: 6, keys.GRIP: 1}),
    action=codecs.absolute_joints_action.override(
        tgt_joints_key=keys.TARGET_JOINTS, tgt_grip_key=keys.TARGET_GRIP, num_joints=6
    ),
    horizon=1.0,
)

# IK variants: reconstruct joint targets from recorded EE targets via IK
joints_ik = ee.override(obs=codecs.joints_obs, action=codecs.ik_joints_action)
joints_ik_sim = joints_ik.override(**{'action.solver': 'lm'})
