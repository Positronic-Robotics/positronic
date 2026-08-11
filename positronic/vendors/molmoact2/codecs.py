"""MolmoAct2 observation codec for the DROID action space (3 cameras + joint/grip state)."""

from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.policy.codec import Codec
from positronic.vendors import molmoact2


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

    def _image(self, key: str, inputs: dict[str, Any]) -> np.ndarray:
        frame = np.asarray(inputs[key])
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Image '{key}' must be HWC with 3 channels, got {frame.shape}")
        return frame

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        joints = np.asarray(inputs[self._joint_key], dtype=np.float32).reshape(-1)
        grip = np.asarray(inputs[self._grip_key], dtype=np.float32).reshape(-1)
        return {
            molmoact2.IMAGES: [self._image(k, inputs) for k in self._cameras],
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

# franka_droid runs at 15 Hz; the model emits a 15-step horizon and compose executes all steps by default.
droid = codecs.compose.override(obs=molmoact2_obs, action=_action, fps=15.0)
