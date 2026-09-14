"""Server-side DROID observation and action conversion for non-commercial G0.5 evaluation."""

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.drivers.roboarm import command
from positronic.policy.codec import ActionTimestamp, Codec
from positronic.vendors.galaxea import protocol


class DroidCodec(Codec):
    """Map canonical 0=open, 1=closed grip to Galaxea's inverted convention in both directions.

    An absent gripper prediction emits no gripper command, preserving the driver's last target.
    Arm predictions are required. The upstream processor owns resizing and normalization.
    """

    def __init__(
        self,
        exterior_camera: str = keys.EXTERIOR_IMAGE,
        wrist_camera: str = keys.WRIST_IMAGE,
        joint_key: str = keys.JOINTS,
        grip_key: str = keys.GRIP,
        task_key: str = keys.TASK,
        fps: float = 15.0,
    ):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        self._cameras = {protocol.EXTERIOR_IMAGE: exterior_camera, protocol.WRIST_IMAGE: wrist_camera}
        self._joint_key = joint_key
        self._grip_key = grip_key
        self._task_key = task_key
        self.fps = fps

    @staticmethod
    def _vector(value, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if size == 1 and array.ndim == 0:
            array = array.reshape(1)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f'{name} must be a finite ({size},) vector, got {array.shape}')
        return array

    @staticmethod
    def _image(value, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape) == 0:
            raise ValueError(f'{name} must be a nonempty uint8 HWC RGB image, got {image.shape}, {image.dtype}')
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    def encode(self, data: dict) -> dict:
        grip = self._vector(data[self._grip_key], 1, self._grip_key)
        if np.any((grip < 0) | (grip > 1)):
            raise ValueError('Observed grip must be in [0, 1]')
        return {
            protocol.IMAGES: {
                **{name: self._image(data[key], key) for name, key in self._cameras.items()},
                protocol.DUMMY_WRIST_RIGHT: np.zeros((3, 224, 224), dtype=np.uint8),
            },
            protocol.STATE: {
                protocol.RIGHT_ARM: self._vector(data[self._joint_key], 7, self._joint_key),
                protocol.RIGHT_GRIPPER: 1.0 - grip,
            },
            protocol.TASK: data[self._task_key],
            protocol.FREQUENCY: self.fps,
            protocol.EMBODIMENT_TYPE: protocol.DROID_FRANKA,
        }

    def _decode_single(self, data: dict) -> dict:
        joints = self._vector(data[protocol.RIGHT_ARM], 7, protocol.RIGHT_ARM)
        result = {keys.ROBOT_COMMAND: command.JointPosition(positions=joints)}
        if protocol.RIGHT_GRIPPER in data:
            grip = self._vector(data[protocol.RIGHT_GRIPPER], 1, protocol.RIGHT_GRIPPER)
            result[keys.TARGET_GRIP] = float(np.clip(1.0 - grip[0], 0.0, 1.0))
        return result


@cfn.config(codec=cfn.Config(DroidCodec))
def droid(codec: DroidCodec):
    return ActionTimestamp(fps=codec.fps) | codecs.droid_execution(action=codec)
