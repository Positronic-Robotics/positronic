"""MolmoAct2 codecs: the DROID action space, and the bimanual YAM one."""

from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.policy.codecs import BimanualJointsAction, Codec, bimanual_joints_state
from positronic.vendors import molmoact2


def _image(key: str, inputs: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(inputs[key])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
    return frame


class MolmoAct2ObservationCodec(Codec):
    """Encodes positronic observations into MolmoAct2 ``predict_action`` inputs.

    Emits the ordered camera list ``[exterior_1, exterior_2, wrist]`` (uint8 HWC RGB), the raw
    8-D state ``[joint_positions(7), grip(1)]``, and the bare language task. MolmoAct2 normalizes
    the state and resizes the images itself.
    """

    def __init__(
        self,
        wrist_camera: str = keys.WRIST_IMAGE,
        exterior_camera_1: str = keys.EXTERIOR_IMAGE,
        exterior_camera_2: str | None = None,
        joint_key: str = keys.JOINTS,
        grip_key: str = keys.GRIP,
    ):
        self._cameras = (exterior_camera_1, exterior_camera_2 or exterior_camera_1, wrist_camera)
        self._joint_key = joint_key
        self._grip_key = grip_key

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        joints = np.asarray(inputs[self._joint_key], dtype=np.float32).reshape(-1)
        grip = np.asarray(inputs[self._grip_key], dtype=np.float32).reshape(-1)
        return {
            molmoact2.IMAGES: [_image(k, inputs) for k in self._cameras],
            molmoact2.STATE: np.concatenate([joints, grip]).astype(np.float32),
            molmoact2.TASK: inputs.get(keys.TASK, ''),
        }


molmoact2_obs = cfn.Config(MolmoAct2ObservationCodec)


# MolmoAct2 returns absolute joint positions (7) + gripper (1) already in raw robot units, so the
# 8-vector decodes straight into a JointPosition command. ``tgt_*_key`` are training-only on
# AbsoluteJointsAction; serving reads the 8-vector directly.
# Its gripper follows the DROID convention (0=open, 1=closed), matching positronic's Robotiq/DH
# drivers, so the grip passes through unchanged on both state-in and target_grip-out.
_action = codecs.absolute_joints_action.override(tgt_joints_key=keys.TARGET_JOINTS, tgt_grip_key=keys.TARGET_GRIP)

# franka_droid training observations are sampled at 15 Hz.
droid = codecs.compose.override(
    obs=molmoact2_obs, action=codecs.droid_execution.override(action=_action), training_fps=15.0
)
droid_3cam = droid.override(**{'obs.exterior_camera_2': keys.EXTERIOR_IMAGE_2})


class MolmoAct2BimanualObservationCodec(Codec):
    """Encodes a bimanual YAM observation into MolmoAct2 ``predict_action`` inputs.

    Cameras go in the order ``[top, left, right]``. The state is 14-D: per arm, 6 joints then the gripper
    width, left arm first.
    """

    def __init__(
        self,
        top_camera: str = keys.EXTERIOR_IMAGE,
        left_camera: str = keys.WRIST_LEFT_IMAGE,
        right_camera: str = keys.WRIST_RIGHT_IMAGE,
    ):
        self._cameras = (top_camera, left_camera, right_camera)

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            molmoact2.IMAGES: [_image(k, inputs) for k in self._cameras],
            molmoact2.STATE: bimanual_joints_state(inputs),
            molmoact2.TASK: inputs.get(keys.TASK, ''),
        }


# The upstream YAM example records and runs at 30 Hz.
yam_bimanual = codecs.compose.override(
    obs=cfn.Config(MolmoAct2BimanualObservationCodec), action=cfn.Config(BimanualJointsAction), training_fps=30.0
)
