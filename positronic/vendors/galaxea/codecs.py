"""Server-side DROID observation and action conversion for non-commercial G0.5 evaluation."""

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg import codecs
from positronic.drivers.roboarm import command
from positronic.policy.codecs import Codec, FlipGrip
from positronic.vendors.galaxea import protocol


class DroidCodec(Codec):
    """Map a DROID observation to G0.5 inputs, and a G0.5 prediction to a joint command and a grip target.

    The grip passes through as it comes, under ``keys.GRIP``: ``droid`` composes ``FlipGrip``, which flips that key
    to Galaxea's aperture (1 = open).
    An absent gripper prediction emits no gripper command, preserving the driver's last target.
    Arm predictions are required. The upstream processor owns resizing and normalization.
    """

    def __init__(
        self,
        exterior_camera: str = keys.EXTERIOR_IMAGE,
        wrist_camera: str = keys.WRIST_IMAGE,
        joint_key: str = keys.JOINTS,
        task_key: str = keys.TASK,
        fps: float = 15.0,
    ):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        self._cameras = {protocol.EXTERIOR_IMAGE: exterior_camera, protocol.WRIST_IMAGE: wrist_camera}
        self._joint_key = joint_key
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
        grip = self._vector(data[keys.GRIP], 1, keys.GRIP)
        if np.any((grip < 0) | (grip > 1)):
            raise ValueError('Observed grip must be in [0, 1]')
        return {
            protocol.IMAGES: {
                **{name: self._image(data[key], key) for name, key in self._cameras.items()},
                protocol.DUMMY_WRIST_RIGHT: np.zeros((3, 224, 224), dtype=np.uint8),
            },
            protocol.STATE: {
                protocol.RIGHT_ARM: self._vector(data[self._joint_key], protocol.RIGHT_ARM_WIDTH, self._joint_key),
                protocol.RIGHT_GRIPPER: grip,
            },
            protocol.TASK: data[self._task_key],
            protocol.FREQUENCY: self.fps,
            protocol.EMBODIMENT_TYPE: protocol.DROID_FRANKA,
        }

    def _decode_single(self, data: dict) -> dict:
        joints = self._vector(data[protocol.RIGHT_ARM], protocol.RIGHT_ARM_WIDTH, protocol.RIGHT_ARM)
        result = {keys.ROBOT_COMMAND: command.JointPosition(positions=joints)}
        if protocol.RIGHT_GRIPPER in data:
            grip = self._vector(data[protocol.RIGHT_GRIPPER], 1, protocol.RIGHT_GRIPPER)
            result[keys.TARGET_GRIP] = float(np.clip(grip[0], 0.0, 1.0))
        return result


@cfn.config(codec=cfn.Config(DroidCodec))
def droid(codec: DroidCodec):
    """``codec`` under DROID impedance, with the grip in Galaxea's aperture."""
    return codecs.droid_execution(action=FlipGrip() | codec)
