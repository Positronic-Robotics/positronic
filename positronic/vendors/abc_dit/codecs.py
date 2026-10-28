"""The server-side codecs of ABC-DiT. They encode a bimanual YAM observation as a request to ABC's policy server, and
decode its 14-D actions as one joint command and one grip target per arm."""

from dataclasses import dataclass
from typing import Any

import configuronic as cfn
import numpy as np

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
# ABC's training marks the prompt of every sim episode with this prefix.
SIM_PROMPT_PREFIX = 'sim '
# The DiT checkpoints predict chunks of this many actions.
CHUNK_LENGTH = 30
# The longest action prefix that `abc_dit_xl_200k_model.pt` trained on. Its sidecar records `max_action_prefix` 8, which
# ABC's training samples as an exclusive bound.
MAX_PREFIX_LENGTH = 7


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


@dataclass(frozen=True)
class ActionPrefix:
    """ABC's real-time chunking for a client that plays `executed` actions of each chunk.

    Each request asks ABC to start the new chunk with the last `length` actions that the client plays of the previous
    chunk, and the reply holds the `executed` actions that come after them.
    """

    length: int
    executed: int

    def __post_init__(self):
        if not 1 <= self.length <= MAX_PREFIX_LENGTH:
            raise ValueError(f'An action prefix holds 1 to {MAX_PREFIX_LENGTH} actions, got {self.length}')
        if not self.length <= self.executed <= CHUNK_LENGTH - self.length:
            raise ValueError(f'The prefix and the played actions must fit in one chunk of {CHUNK_LENGTH}, got {self}')


def _channels_first_view(frame: Any, key: str, box: CropBox) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
    width, height = VIEW_SIZE
    frame = image.transform_frame(box.crop(frame), width, height, mode=image.ResizeMode.PAD)
    return np.ascontiguousarray(frame.transpose(2, 0, 1))


# The session metadata fields that name what the codec does: the crop box of each view, in fractions of the frame,
# the prompt prefix, and the length of the action prefix (0 for none).
CROPS_META = 'abc_dit.crops'
PROMPT_PREFIX_META = 'abc_dit.prompt_prefix'
PREFIX_LENGTH_META = 'abc_dit.prefix_length'


class AbcObservationCodec(Codec):
    """Encodes a bimanual YAM observation as a request to ABC's policy server.

    Each view is cropped to its box in ``crops``, keyed by the observation's image key, and then goes as a
    channels-first ``uint8`` RGB image under its ABC camera name. A view with no box goes whole. The state is 14-D:
    per arm, 6 joints then the grip, left arm first. ``FlipGrip`` turns the grip into the gripper width that ABC
    reads. ``prompts`` maps a task instruction to the prompt that the checkpoint trained under, after
    ``prompt_prefix``, and an instruction it does not hold is refused. With
    ``action_prefix``, each request also carries ABC's prefix length and the number of actions the client plays.
    """

    def __init__(
        self,
        prompts: dict[str, str],
        prompt_prefix: str = '',
        crops: dict[str, CropBox] | None = None,
        top_camera: str = keys.EXTERIOR_IMAGE,
        left_camera: str = keys.WRIST_LEFT_IMAGE,
        right_camera: str = keys.WRIST_RIGHT_IMAGE,
        action_prefix: ActionPrefix | None = None,
    ):
        self._prompt_prefix = prompt_prefix
        self._action_prefix = action_prefix
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
        request = {
            abc_dit.STATE: bimanual_joints_state(inputs),
            abc_dit.IMAGES: {
                name: _channels_first_view(inputs[key], key, self._crops.get(key, FULL_FRAME))
                for name, key in self._cameras.items()
            },
            abc_dit.PROMPT: self._prompts[task],
        }
        if self._action_prefix is not None:
            request[abc_dit.PREFIX_LENGTH] = self._action_prefix.length
            request[abc_dit.EXECUTED_ACTIONS] = self._action_prefix.executed
        return request

    @property
    def meta(self) -> dict[str, Any]:
        prefix = self._action_prefix
        return {
            CROPS_META: {key: [box.left, box.top, box.right, box.bottom] for key, box in self._crops.items()},
            PROMPT_PREFIX_META: self._prompt_prefix,
            PREFIX_LENGTH_META: 0 if prefix is None else prefix.length,
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

# The yambox ZED views (960x600, focal length 367 px) cut to 4:3 with the field of view of ABC's D405 views.
# README.md gives the reason for each box.
YAMBOX_CROPS = {
    keys.EXTERIOR_IMAGE: CropBox(4 / 15, 0.32, 5 / 6, 1.0),
    keys.WRIST_LEFT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
    keys.WRIST_RIGHT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
}

# The whole exterior frame, where the arm bases lie 0.48 of the width apart (ABC's top view: 0.47).
YAMBOX_WIDE_TOP_CROPS = {**YAMBOX_CROPS, keys.EXTERIOR_IMAGE: FULL_FRAME}

# The crop sets that a session can name.
CROPS = {'none': {}, 'yambox': YAMBOX_CROPS, 'yambox_wide_top': YAMBOX_WIDE_TOP_CROPS}


@cfn.config(training_fps=FPS, prompt_prefix=SIM_PROMPT_PREFIX, crops=None, action_prefix=None)
def multitask_200k(
    training_fps: float, prompt_prefix: str, crops: dict[str, CropBox] | None, action_prefix: ActionPrefix | None
):
    """The codec of the multi-task checkpoint, with each arm's grip in the gripper width (1 = open) that ABC speaks.
    ``prompt_prefix``, ``crops`` and ``action_prefix`` go to ``AbcObservationCodec``."""
    obs = AbcObservationCodec(
        prompts=MULTITASK_200K_PROMPTS, prompt_prefix=prompt_prefix, crops=crops, action_prefix=action_prefix
    )
    pair = codecs.compose(obs=obs, action=BimanualJointsAction(num_joints=YAM_JOINTS), training_fps=training_fps)
    return FlipGrip(arms=keys.BIMANUAL_ARMS) | pair
