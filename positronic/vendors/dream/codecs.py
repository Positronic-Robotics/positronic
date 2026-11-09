"""The server-side codec of Dream-1. It encodes a bimanual YAM observation as a YAM station observation of DreamAPI,
and decodes each action row as one joint command and one grip target per arm."""

from collections.abc import Mapping
from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.dataset.transforms import image
from positronic.drivers.roboarm.models import YAM_JOINTS
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import BimanualJointsAction, Codec, FlipGrip, bimanual_joints_state
from positronic.vendors import dream

# Each row of a Dream-1 chunk is one step of a 20 Hz clock, the rate of the Dream SDK station.
FPS = 20.0
# Dream-1 reads each frame at this (width, height).
FRAME_SIZE = (320, 176)
# Dream-1 reads the last 9 frames of each camera, 0.2 s apart.
WINDOW_FRAMES = 9
FRAME_PERIOD_S = 0.2

# The session metadata field that states whether the codec cuts each frame to the aspect ratio of `FRAME_SIZE`.
CROP_META = 'dream.crop_to_aspect'


def rows_ahead(inputs: Mapping[str, Any]) -> int:
    """How many rows of its current chunk the client plays after the observation of `inputs`, rounded to whole rows,
    from the plan that it sends (`policy_keys.RTC.ACTION_PREFIX`). With no plan, no row is ahead."""
    prefix = inputs.get(policy_keys.RTC.ACTION_PREFIX)
    if prefix is None:
        return 0
    rows = len(prefix[policy_keys.RTC.PREFIX_ACTIONS]) + prefix[policy_keys.RTC.OBSERVATION_OFFSET_SEC] * FPS
    return max(0, round(rows))


def rows_queued(inputs: Mapping[str, Any]) -> int:
    """How many rows of its plan the client had not started at the observation of `inputs`. The row before them is
    the one that the client commanded at the observation. With no plan, no row is queued."""
    prefix = inputs.get(policy_keys.RTC.ACTION_PREFIX)
    if prefix is None:
        return 0
    rows = len(prefix[policy_keys.RTC.PREFIX_ACTIONS])
    started = prefix[policy_keys.RTC.OBSERVATION_OFFSET_SEC] <= 0
    return max(0, rows - 1) if started else rows


class DreamObservationCodec(Codec):
    """Encodes a bimanual YAM observation as the YAM station observation of DreamAPI.

    Each camera is a frame stack, oldest frame first, or one frame. Each frame goes to `FRAME_SIZE` as RGB `uint8`.
    With `crop_to_aspect`, a centred crop to the aspect ratio of `FRAME_SIZE` comes first. The grip of each arm must
    be the gripper opening (1 = open), so `FlipGrip(arms=...)` composes to the left. `policy_keys.REPLAN_NOW` passes
    through. `dream.ROWS_AHEAD` holds `rows_ahead` of the inputs, and `dream.ROWS_QUEUED` holds `rows_queued`.
    """

    def __init__(
        self,
        crop_to_aspect: bool = True,
        top_camera: str = keys.EXTERIOR_IMAGE,
        left_camera: str = keys.WRIST_LEFT_IMAGE,
        right_camera: str = keys.WRIST_RIGHT_IMAGE,
    ):
        self._crop_to_aspect = crop_to_aspect
        self._cameras = {dream.TOP_CAMERA: top_camera, dream.LEFT_CAMERA: left_camera, dream.RIGHT_CAMERA: right_camera}

    @staticmethod
    def _centre_crop(frame: np.ndarray, aspect: float) -> np.ndarray:
        """The largest centred part of `frame` with a width-to-height ratio of `aspect`."""
        height, width = frame.shape[:2]
        if width / height > aspect:
            cut = round(height * aspect)
            left = (width - cut) // 2
            return frame[:, left : left + cut]
        cut = round(width / aspect)
        top = (height - cut) // 2
        return frame[top : top + cut]

    def _frame(self, frame: np.ndarray, key: str) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
        if self._crop_to_aspect:
            frame = self._centre_crop(frame, FRAME_SIZE[0] / FRAME_SIZE[1])
        return image.transform_frame(frame, *FRAME_SIZE, mode=image.ResizeMode.STRETCH)

    def _window(self, value: Any, key: str) -> np.ndarray:
        stack = np.asarray(value)
        if stack.ndim == 3:
            stack = stack[None]
        return np.stack([self._frame(frame, key) for frame in stack[-WINDOW_FRAMES:]]).astype(np.uint8)

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        state = bimanual_joints_state(inputs).reshape(len(keys.BIMANUAL_ARMS), YAM_JOINTS + 1)
        observation: dict[str, Any] = {dream.PROMPT: inputs.get(keys.TASK, ''), **dream.STATION_ROBOT}
        for arm, arm_state in zip(keys.BIMANUAL_ARMS, state, strict=True):
            slot = dream.ARM_SLOTS[arm]
            observation[dream.JOINT_POS.format(slot=slot)] = arm_state[:YAM_JOINTS].tolist()
            observation[dream.GRIPPER_POS.format(slot=slot)] = arm_state[YAM_JOINTS:].tolist()
        frames = {
            dream.FRAME_WINDOW.format(camera=camera): self._window(inputs[key], key)
            for camera, key in self._cameras.items()
        }
        replan = bool(inputs.get(policy_keys.REPLAN_NOW, False))
        return {
            dream.OBSERVATION: observation,
            dream.FRAMES: frames,
            policy_keys.REPLAN_NOW: replan,
            dream.ROWS_AHEAD: rows_ahead(inputs),
            dream.ROWS_QUEUED: rows_queued(inputs),
        }

    @property
    def meta(self) -> dict[str, Any]:
        return {CROP_META: self._crop_to_aspect}


@cfn.config(training_fps=FPS, crop_to_aspect=True)
def yam_bimanual(training_fps: float, crop_to_aspect: bool):
    """The codec of Dream-1 on a bimanual YAM: each grip goes to Dream-1 as the gripper opening (1 = open)."""
    pair = codecs.compose(
        obs=DreamObservationCodec(crop_to_aspect=crop_to_aspect),
        action=BimanualJointsAction(num_joints=YAM_JOINTS),
        training_fps=training_fps,
    )
    return FlipGrip(arms=keys.BIMANUAL_ARMS) | pair
