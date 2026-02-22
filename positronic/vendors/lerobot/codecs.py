"""LeRobot codecs (observation encoder | action decoder pairs)."""

from positronic.cfg import codecs

ee = codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action)
joints = ee.override(obs=codecs.joints_obs)

# Trajectory variants: use actual robot trajectory as action target instead of commanded targets
ee_traj = ee.override(action=codecs.traj_ee_action, binarize_grip=('grip',))

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_traj = codecs.compose.override(
    obs=codecs.joints_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip'),
    binarize_grip=('grip',),
)
