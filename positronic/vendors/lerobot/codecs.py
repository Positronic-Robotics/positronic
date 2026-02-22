"""LeRobot codecs (observation encoder | action decoder pairs)."""

from positronic.cfg import codecs

eepose_absolute = codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action)
joints_absolute = eepose_absolute.override(obs=codecs.joints_obs)

# Trajectory variants: use actual robot trajectory as action target instead of commanded targets
_traj = {
    'action.tgt_ee_pose_key': 'robot_state.ee_pose',
    'action.tgt_grip_key': 'grip',
    'binarize_grip_keys': ('grip',),
}
eepose_absolute_traj = eepose_absolute.override(**_traj)
joints_absolute_traj = joints_absolute.override(**_traj)
