"""LeRobot codecs for lerobot 0.4.x (512x512 images)."""

from positronic.cfg import codecs

ee = codecs.compose.override(obs=codecs.eepose_obs.override(image_size=(512, 512)), action=codecs.absolute_pos_action)
joints = codecs.compose.override(
    obs=codecs.joints_obs.override(image_size=(512, 512)), action=codecs.absolute_pos_action
)
