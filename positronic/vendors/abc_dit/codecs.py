"""The server-side codecs of ABC-DiT. They encode a bimanual YAM observation as a request to ABC's policy server, and
decode its 14-D actions as one joint command and one grip target per arm."""

from dataclasses import dataclass
from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.dataset.transforms import image
from positronic.policy.codecs import BimanualJointsAction, Codec, bimanual_joints_state
from positronic.vendors import abc_dit

# ABC's policies learn from 30 Hz recordings, and each action is one step of that clock.
FPS = 30.0
# ABC renders its sim training views at this (width, height). Its DiT then pads each view to a square.
VIEW_SIZE = (224, 168)
# Per arm, the state and each action hold the joints of a YAM arm and then the gripper width.
YAM_JOINTS = 6
BIMANUAL_SIZE = (YAM_JOINTS + 1) * len(keys.BIMANUAL_ARMS)
# ABC's training marks the prompt of every sim episode with this prefix.
SIM_PROMPT_PREFIX = 'sim '


@dataclass(frozen=True)
class CropBox:
    """A box in fractions of a frame's width and height: 0 is the left or top edge, 1 the right or bottom edge."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        if not (0.0 <= self.left < self.right <= 1.0 and 0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError(f'A crop box must lie inside the frame and hold an area, got {self}')

    def crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        rows = slice(round(self.top * height), round(self.bottom * height))
        columns = slice(round(self.left * width), round(self.right * width))
        return frame[rows, columns]


FULL_FRAME = CropBox(0.0, 0.0, 1.0, 1.0)


def _channels_first_view(frame: Any, key: str, box: CropBox) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
    width, height = VIEW_SIZE
    frame = image.transform_frame(box.crop(frame), width, height, mode=image.ResizeMode.PAD)
    return np.ascontiguousarray(frame.transpose(2, 0, 1))


class AbcObservationCodec(Codec):
    """Encodes a bimanual YAM observation as a request to ABC's policy server.

    Each view is cropped to its box in ``crops``, keyed by the observation's image key, and then goes as a
    channels-first ``uint8`` RGB image under its ABC camera name. A view with no box goes whole. The state is 14-D:
    per arm, 6 joints then the gripper width, left arm first. ``prompts`` maps a task instruction to the prompt that
    the checkpoint trained under, after ``prompt_prefix``, and an instruction it does not hold is refused.
    """

    def __init__(
        self,
        prompts: dict[str, str],
        prompt_prefix: str = '',
        crops: dict[str, CropBox] | None = None,
        top_camera: str = keys.EXTERIOR_IMAGE,
        left_camera: str = keys.WRIST_LEFT_IMAGE,
        right_camera: str = keys.WRIST_RIGHT_IMAGE,
    ):
        self._prompts = {task: prompt_prefix + prompt for task, prompt in prompts.items()}
        self._cameras = {
            abc_dit.TOP_CAMERA: top_camera,
            abc_dit.LEFT_CAMERA: left_camera,
            abc_dit.RIGHT_CAMERA: right_camera,
        }
        self._crops = dict(crops or {})
        if unknown := set(self._crops) - set(self._cameras.values()):
            raise ValueError(f'Crop boxes name no camera of this codec: {sorted(unknown)}')

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task = inputs.get(keys.TASK, '')
        if task not in self._prompts:
            raise ValueError(f'The checkpoint has no prompt for the task {task!r}; it knows {sorted(self._prompts)}')
        return {
            abc_dit.STATE: bimanual_joints_state(inputs),
            abc_dit.IMAGES: {
                name: _channels_first_view(inputs[key], key, self._crops.get(key, FULL_FRAME))
                for name, key in self._cameras.items()
            },
            abc_dit.PROMPT: self._prompts[task],
        }


# The prompt that each sim task's episodes carried in the training of `abc_dit_xl_200k_model.pt`, after
# `SIM_PROMPT_PREFIX`, keyed by the task instruction that ABC's sim reports. The release's
# `abc_dit_xl_200k_model.json` records the same map by task name.
MULTITASK_200K_PROMPTS = {
    'put the plastic bottles in the bin': 'throw plastic bottles in bin',
    'sweep away paper scraps from the table': 'sweep away paper scraps from the table',
    'turn mug right side up': 'turn mug right side up',
    'load plates into tabletop dish rack': 'load plates into tabletop dish rack',
    'hang mug on mug rack': 'hang the mug on the mug rack',
}

# The views of the yambox, cropped to 4:3 like ABC's D405 views. Its ZED cameras give 960x600 frames with a focal
# length of 367 px. The top crop is 544x408 px: a 58 degree vertical field of view, as ABC's top view has, over the
# arms and the table. Each wrist crop is the middle 800x600 px, where the open fingertips lie as far apart, as a
# share of the width, as in ABC's wrist views.
YAMBOX_CROPS = {
    keys.EXTERIOR_IMAGE: CropBox(4 / 15, 0.32, 5 / 6, 1.0),
    keys.WRIST_LEFT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
    keys.WRIST_RIGHT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
}

multitask_200k = codecs.compose.override(
    obs=cfn.Config(AbcObservationCodec, prompts=MULTITASK_200K_PROMPTS, prompt_prefix=SIM_PROMPT_PREFIX),
    action=cfn.Config(BimanualJointsAction, num_joints=YAM_JOINTS),
    training_fps=FPS,
)
multitask_200k_yambox_crop = multitask_200k.override(**{'obs.crops': YAMBOX_CROPS})
multitask_200k_yambox_crop_no_sim_prefix = multitask_200k_yambox_crop.override(**{'obs.prompt_prefix': ''})
