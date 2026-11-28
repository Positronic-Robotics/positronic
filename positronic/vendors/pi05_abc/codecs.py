"""The server-side codec of the pi0.5 ABC bottles checkpoint. It encodes a bimanual YAM observation as a request to
the author's server, and decodes its 14-D actions as one joint command and one grip target per arm."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.dataset.transforms import image
from positronic.policy.codecs import BimanualJointsAction, Codec, FlipGrip, bimanual_joints_state
from positronic.utils import check_rgb_frame
from positronic.vendors import pi05_abc

# ABC's recordings run at 30 Hz, and each action is one step of that clock.
FPS = 30.0
# ABC's cameras record 4:3 views; the codec sends each one at this (width, height), and openpi pads it to a square.
VIEW_SIZE = (224, 168)
# Per arm, the state and each action hold the joints of a YAM arm and then the gripper width.
YAM_JOINTS = 6
BIMANUAL_SIZE = (YAM_JOINTS + 1) * len(keys.BIMANUAL_ARMS)
# The one task of the checkpoint, and the prompt it trained under: the task folder name of the ABC-130k release.
TASK = 'put the plastic bottles in the bin'
# The session metadata field that names the crop box of each view, in fractions of the frame.
CROPS_META = 'pi05_abc.crops'


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


class BottlesObservationCodec(Codec):
    """Encodes a bimanual YAM observation as a request to the author's server.

    Each view is cropped to its box in ``crops``, keyed by the observation's image key, and then goes as a
    channels-first ``uint8`` RGB image under its camera role. A view with no box goes whole. The state is 14-D: per
    arm, 6 joints then the grip, left arm first. A task other than ``TASK`` is refused.
    """

    CAMERAS = {
        pi05_abc.TOP_CAMERA: keys.EXTERIOR_IMAGE,
        pi05_abc.LEFT_CAMERA: keys.WRIST_LEFT_IMAGE,
        pi05_abc.RIGHT_CAMERA: keys.WRIST_RIGHT_IMAGE,
    }

    def __init__(self, crops: dict[str, CropBox] | None = None):
        self._crops = dict(crops or {})
        if unknown := set(self._crops) - set(self.CAMERAS.values()):
            raise ValueError(f'Crop boxes name no camera of this codec: {sorted(unknown)}')

    @staticmethod
    def _channels_first_view(frame: Any, key: str, box: CropBox) -> np.ndarray:
        frame = np.asarray(frame)
        check_rgb_frame(frame, f"image '{key}'")
        width, height = VIEW_SIZE
        frame = image.transform_frame(box.crop(frame), width, height, mode=image.ResizeMode.PAD)
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task = inputs.get(keys.TASK, '')
        if task != TASK:
            raise ValueError(f'The checkpoint knows only the task {TASK!r}, not {task!r}')
        return {
            pi05_abc.STATE: bimanual_joints_state(inputs),
            pi05_abc.IMAGES: {
                role: self._channels_first_view(inputs[key], key, self._crops.get(key, FULL_FRAME))
                for role, key in self.CAMERAS.items()
            },
            pi05_abc.PROMPT: TASK,
        }

    @property
    def meta(self) -> dict[str, Any]:
        return {CROPS_META: {key: [box.left, box.top, box.right, box.bottom] for key, box in self._crops.items()}}


# The yambox ZED views (960x600, focal length 367 px) cut to 4:3 with the field of view of ABC's D405 views.
# README.md gives the reason for each box.
YAMBOX_CROPS = {
    keys.EXTERIOR_IMAGE: CropBox(4 / 15, 0.32, 5 / 6, 1.0),
    keys.WRIST_LEFT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
    keys.WRIST_RIGHT_IMAGE: CropBox(1 / 12, 0.0, 11 / 12, 1.0),
}

# The whole exterior frame, where the arm bases lie 0.48 of the width apart (ABC's top view: 0.47).
YAMBOX_WIDE_TOP_CROPS = {**YAMBOX_CROPS, keys.EXTERIOR_IMAGE: FULL_FRAME}


class Crop(StrEnum):
    """The crop sets that a session can name."""

    NONE = 'none'
    YAMBOX = 'yambox'
    YAMBOX_WIDE_TOP = 'yambox_wide_top'


CROPS = {Crop.NONE: {}, Crop.YAMBOX: YAMBOX_CROPS, Crop.YAMBOX_WIDE_TOP: YAMBOX_WIDE_TOP_CROPS}


@cfn.config(crops=None)
def bottles(crops: dict[str, CropBox] | None):
    """The codec of the checkpoint, with each arm's grip in the gripper width (1 = open) that it speaks."""
    obs = BottlesObservationCodec(crops=crops)
    pair = codecs.compose(obs=obs, action=BimanualJointsAction(num_joints=YAM_JOINTS), training_fps=FPS)
    return FlipGrip(arms=keys.BIMANUAL_ARMS) | pair
