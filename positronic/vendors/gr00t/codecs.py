"""DROID observations and joint-position actions for GR00T."""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image

from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Get
from positronic.drivers.roboarm import command, models
from positronic.policy import keys as policy_keys
from positronic.policy.codec import (
    ACTION,
    GR00T_MODALITY,
    LEROBOT_FEATURES,
    BinarizeGripInference,
    ChangeEEFrame,
    Codec,
    Metadata,
    lerobot_action,
    lerobot_image,
    lerobot_vector,
    warm_image,
    warm_pose,
    warm_vector,
)
from positronic.vendors import gr00t


class DroidCodec(Codec):
    """Encode poses in the DROID tool frame and decode upstream's absolute joint targets.

    Training uses recorded pose/joint/gripper trajectories as absolute action labels. GR00T's
    checkpoint processor converts those labels to relative actions and back during inference.
    """

    # Matches GR00T's gr00t/data/state_action/droid_frame.py; row-based rot6d follows this correction.
    _ROTATION_CORRECTION = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)

    def __init__(self, image_mappings: dict[str, str]):
        self.image_mappings = dict(image_mappings)

    @classmethod
    def _encode_pose(cls, value):
        pose = geom.Transform3D.from_vector(np.asarray(value), geom.Rotation.Representation.QUAT)
        rotation = pose.rotation.as_rotation_matrix @ cls._ROTATION_CORRECTION
        return np.concatenate([pose.translation, rotation[:2].reshape(6)]).astype(np.float32)

    @staticmethod
    def _encode_image(frame):
        return image.resize_with_pad_per_frame(*gr00t.IMAGE_SIZE, Image.Resampling.BILINEAR, np.asarray(frame))

    def warm_inputs(self, task: str) -> dict[str, Any]:
        """The rig-side inputs a warm carries, under the names this codec reads."""
        frame = warm_image(*gr00t.IMAGE_SIZE)
        return {
            keys.TASK: task,
            **dict.fromkeys(self.image_mappings.values(), frame),
            keys.EE_POSE: warm_pose(),
            keys.GRIP: warm_vector(1),
            keys.JOINTS: warm_vector(gr00t.STATE_DIMS[gr00t.JOINT_POSITION]),
        }

    def encode(self, inputs: dict) -> dict:
        state = {
            gr00t.EE_POSE: self._encode_pose(inputs[keys.EE_POSE]),
            gr00t.GRIP: np.asarray(inputs[keys.GRIP], dtype=np.float32).reshape(1),
            gr00t.JOINT_POSITION: np.asarray(inputs[keys.JOINTS], dtype=np.float32).reshape(
                gr00t.STATE_DIMS[gr00t.JOINT_POSITION]
            ),
        }
        return {
            gr00t.VIDEO: {
                name: self._encode_image(inputs[source])[None, None] for name, source in self.image_mappings.items()
            },
            gr00t.STATE: {name: value[None, None] for name, value in state.items()},
            gr00t.LANGUAGE: {gr00t.TASK: [[inputs[keys.TASK]]]},
        }

    def _decode_single(self, data: dict) -> dict:
        return {
            keys.ROBOT_COMMAND: command.JointPosition(
                positions=np.asarray(data[gr00t.JOINT_POSITION]).reshape(gr00t.STATE_DIMS[gr00t.JOINT_POSITION]),
                mode=DROID_IMPEDANCE,
            ),
            keys.TARGET_GRIP: np.asarray(data[gr00t.GRIP]).item(),
        }

    def _derive_pose(self, episode: Episode):
        return tf.Elementwise(episode[keys.EE_POSE], tf.lazy_sequence(self._encode_pose))

    @staticmethod
    def _derive_grip(episode: Episode):
        return tf.Elementwise(episode[keys.GRIP], lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1))

    @staticmethod
    def _derive_image(source: str, episode: Episode):
        return image.resize_with_pad(*gr00t.IMAGE_SIZE, signal=episode[source])

    @property
    def training_encoder(self):
        state_encoders = {
            gr00t.EE_POSE: self._derive_pose,
            gr00t.GRIP: self._derive_grip,
            gr00t.JOINT_POSITION: lambda episode: tf.Elementwise(
                episode[keys.JOINTS], partial(np.asarray, dtype=np.float32)
            ),
        }
        state_meta = {
            name: {gr00t.START: 0, gr00t.END: gr00t.STATE_DIMS[name], gr00t.ORIGINAL_KEY: name}
            for name in state_encoders
        }
        action_meta = {}
        start = 0
        for name in state_encoders:
            dim = gr00t.STATE_DIMS[name]
            action_meta[name] = {gr00t.START: start, gr00t.END: start + dim}
            start += dim
        meta = {
            GR00T_MODALITY: {
                gr00t.STATE: state_meta,
                ACTION: action_meta,
                gr00t.VIDEO: {name: {gr00t.ORIGINAL_KEY: name} for name in self.image_mappings},
                gr00t.ANNOTATION: {
                    gr00t.TASK.removeprefix(gr00t.ANNOTATION + '.'): {gr00t.ORIGINAL_KEY: gr00t.TASK_INDEX}
                },
            },
            LEROBOT_FEATURES: {
                **{name: lerobot_vector(gr00t.STATE_DIMS[name]) for name in state_encoders},
                **{name: lerobot_image(*gr00t.IMAGE_SIZE) for name in self.image_mappings},
                ACTION: lerobot_action(start),
            },
        }
        return Derive(
            meta=meta,
            **{
                **state_encoders,
                keys.TASK: Get(keys.TASK, ''),
                ACTION: lambda episode: tf.concat(
                    *(derive(episode) for derive in state_encoders.values()), dtype=np.float32
                ),
                **{name: partial(self._derive_image, source) for name, source in self.image_mappings.items()},
            },
        )

    @property
    def meta(self):
        return {self.IMAGE_SIZES: dict.fromkeys(self.image_mappings.values(), gr00t.IMAGE_SIZE)}


@cfn.config(
    image_mappings={gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE: keys.WRIST_IMAGE},
    ee_frame=models.DROID_EE_FRAME,
)
def droid(image_mappings: dict[str, str], ee_frame: geom.Transform3D, training_fps: float = 15.0):
    """DROID data conversion and training cadence metadata."""
    return (
        Metadata({policy_keys.ACTION_FPS: training_fps})
        | BinarizeGripInference()
        | ChangeEEFrame(ee_frame)
        | DroidCodec(image_mappings)
    )


droid_three_cameras = droid.override(
    image_mappings={
        gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE,
        gr00t.EXTERIOR_IMAGE_2: keys.EXTERIOR_IMAGE_2,
        gr00t.WRIST_IMAGE: keys.WRIST_IMAGE,
    }
)
