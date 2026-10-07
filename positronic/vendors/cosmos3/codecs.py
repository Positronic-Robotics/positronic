"""The server-side DROID codecs of Cosmos3-Nano. They encode an observation as a request to NVIDIA's server, and
decode its actions as commands."""

import configuronic as cfn

from positronic import keys
from positronic.cfg import codecs
from positronic.policy.observation import ObservationCodec
from positronic.vendors import cosmos3

# The policy learned from DROID recordings at 15 Hz, and it predicts actions at that rate.
FPS = 15.0
# Each view has this (width, height). NVIDIA's server puts the exterior views at half size below the wrist view,
# which gives the 640x540 frame the policy reads.
VIEW_SIZE = (640, 360)

_views_3cam = {
    cosmos3.WRIST_IMAGE: (keys.WRIST_IMAGE, VIEW_SIZE),
    cosmos3.EXTERIOR_IMAGE_1: (keys.EXTERIOR_IMAGE, VIEW_SIZE),
    cosmos3.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE_2, VIEW_SIZE),
}
_obs_3cam = cfn.Config(
    ObservationCodec,
    state={cosmos3.JOINT_POSITION: {keys.JOINTS: 7}, cosmos3.GRIPPER_POSITION: {keys.GRIP: 1}},
    images=_views_3cam,
    task_field=cosmos3.PROMPT,
)
# Each action holds seven absolute joint positions and the gripper's closed fraction, as DROID records them.
_action = codecs.droid_execution.override(
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP)
)

# This codec sends both exterior views, as the policy learned from them. NVIDIA's RoboLab client closes the
# gripper above 0.5, and the codec does the same.
droid_3cam = codecs.compose.override(obs=_obs_3cam, action=_action, training_fps=FPS, binarize_grip=(keys.GRIP,))
# This codec sends one exterior view in both exterior slots, for an eval that renders only one.
droid = droid_3cam.override(
    obs=_obs_3cam.override(images={**_views_3cam, cosmos3.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE, VIEW_SIZE)})
)
