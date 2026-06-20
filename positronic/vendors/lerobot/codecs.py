"""LeRobot codecs for lerobot 0.4.x (512x512 images)."""

from positronic.cfg import codecs

ee = codecs.compose.override(
    obs=codecs.eepose_obs.override(image_size=(512, 512)), action=codecs.absolute_pos_action, horizon=1.0
)
joints = codecs.compose.override(
    obs=codecs.joints_obs.override(image_size=(512, 512)), action=codecs.absolute_pos_action, horizon=1.0
)

# IK variants: reconstruct joint targets from recorded EE targets via IK
joints_ik = ee.override(obs=codecs.joints_obs.override(image_size=(512, 512)), action=codecs.ik_joints_action)
joints_ik_sim = joints_ik.override(**{'action.solver': 'lm'})
