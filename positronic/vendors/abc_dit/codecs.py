"""The server-side codecs of ABC-DiT. They encode a bimanual YAM observation as a request to ABC's policy server, and
decode its 14-D actions as one joint command and one grip target per arm."""

from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic import keys
from positronic.cfg import codecs
from positronic.dataset.transforms import image
from positronic.policy.codecs import BimanualJointsAction, Codec, FlipGrip, bimanual_joints_state
from positronic.vendors import abc_dit

# ABC's policies learn from 30 Hz recordings, and each action is one step of that clock.
FPS = 30.0
# ABC renders its sim training views at this (width, height). Its DiT then pads each view to a square.
VIEW_SIZE = (224, 168)
# Per arm, the state and each action hold the joints of a YAM arm and then the gripper width.
YAM_JOINTS = 6
BIMANUAL_SIZE = (YAM_JOINTS + 1) * len(keys.BIMANUAL_ARMS)

# The prompt that each sim task's episodes carried in the training of `abc_dit_xl_200k_model.pt`, keyed by the task
# instruction that ABC's sim reports. The release's `abc_dit_xl_200k_model.json` records the same map by task name.
MULTITASK_200K_PROMPTS = {
    'put the plastic bottles in the bin': 'sim throw plastic bottles in bin',
    'sweep away paper scraps from the table': 'sim sweep away paper scraps from the table',
    'turn mug right side up': 'sim turn mug right side up',
    'load plates into tabletop dish rack': 'sim load plates into tabletop dish rack',
    'hang mug on mug rack': 'sim hang the mug on the mug rack',
}


def _channels_first_view(key: str, inputs: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(inputs[key])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
    width, height = VIEW_SIZE
    frame = image.resize_with_pad_per_frame(width, height, PilImage.Resampling.BILINEAR, frame)
    return np.ascontiguousarray(frame.transpose(2, 0, 1))


class AbcObservationCodec(Codec):
    """Encodes a bimanual YAM observation as a request to ABC's policy server.

    Each view goes as a channels-first ``uint8`` RGB image under its ABC camera name. The state is 14-D: per arm, 6
    joints then the grip, left arm first. ``FlipGrip`` turns the grip into the gripper width that ABC reads.
    ``prompts`` maps a task instruction to the prompt that the checkpoint trained under, and an instruction it does
    not hold is refused.
    """

    def __init__(
        self,
        prompts: dict[str, str],
        top_camera: str = keys.EXTERIOR_IMAGE,
        left_camera: str = keys.WRIST_LEFT_IMAGE,
        right_camera: str = keys.WRIST_RIGHT_IMAGE,
    ):
        self._prompts = dict(prompts)
        self._cameras = {
            abc_dit.TOP_CAMERA: top_camera,
            abc_dit.LEFT_CAMERA: left_camera,
            abc_dit.RIGHT_CAMERA: right_camera,
        }

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task = inputs.get(keys.TASK, '')
        if task not in self._prompts:
            raise ValueError(f'The checkpoint has no prompt for the task {task!r}; it knows {sorted(self._prompts)}')
        return {
            abc_dit.STATE: bimanual_joints_state(inputs),
            abc_dit.IMAGES: {name: _channels_first_view(key, inputs) for name, key in self._cameras.items()},
            abc_dit.PROMPT: self._prompts[task],
        }


@cfn.config(training_fps=FPS)
def multitask_200k(training_fps: float):
    """The codec of the multi-task checkpoint, with each arm's grip in the gripper width (1 = open) that ABC speaks."""
    pair = codecs.compose(
        obs=AbcObservationCodec(prompts=MULTITASK_200K_PROMPTS),
        action=BimanualJointsAction(num_joints=YAM_JOINTS),
        training_fps=training_fps,
    )
    return FlipGrip(arms=keys.BIMANUAL_ARMS) | pair
