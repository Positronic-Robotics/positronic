"""The server-side DROID codecs of FLUX 3 Action. They encode an observation as a request to BFL's server, and
decode its actions as commands."""

import configuronic as cfn

from positronic import keys
from positronic.cfg import codecs
from positronic.policy.codecs.observation import ObservationCodec
from positronic.vendors import flux3_action

# The policy learned from DROID recordings at 15 Hz, and it predicts actions at that rate.
FPS = 15.0
# Each view has this (width, height). BFL's server refuses a wrist view of another size.
VIEW_SIZE = (640, 360)

_views_3cam = {
    flux3_action.WRIST_IMAGE: (keys.WRIST_IMAGE, VIEW_SIZE),
    flux3_action.EXTERIOR_IMAGE_1: (keys.EXTERIOR_IMAGE, VIEW_SIZE),
    flux3_action.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE_2, VIEW_SIZE),
}
_obs_3cam = cfn.Config(
    ObservationCodec,
    state={flux3_action.JOINT_POSITION: {keys.JOINTS: 7}, flux3_action.GRIPPER_POSITION: {keys.GRIP: 1}},
    images=_views_3cam,
    task_field=flux3_action.PROMPT,
)
# Each action holds seven absolute joint positions and the gripper's closed fraction, as DROID records them.
_action = codecs.droid_execution.override(
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP)
)

# This codec sends both exterior views, as the policy learned from them.
droid_3cam = codecs.compose.override(obs=_obs_3cam, action=_action, training_fps=FPS)
# This codec sends one exterior view in both exterior slots, for an eval that renders only one.
droid = droid_3cam.override(
    obs=_obs_3cam.override(images={**_views_3cam, flux3_action.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE, VIEW_SIZE)})
)
