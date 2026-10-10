"""DROID observations and joint-position actions for GR00T."""

from functools import partial

import configuronic as cfn
import numpy as np
from PIL import Image

from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode, select_timeline
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Get
from positronic.drivers.roboarm import command, models
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import (
    ACTION,
    GR00T_MODALITY,
    LEROBOT_FEATURES,
    BinarizeGripInference,
    ChangeEEFrame,
    Codec,
    Metadata,
    PackObservationFields,
    UnpackActionChunk,
    lerobot_action,
    lerobot_image,
    lerobot_vector,
)
from positronic.policy.codecs.geometry import ConvertPose
from positronic.vendors import gr00t


class DroidCodec(Codec):
    """Prepare DROID state vectors and images, and decode absolute joint targets.

    Training uses recorded pose/joint/gripper trajectories as absolute action labels. GR00T's
    checkpoint processor converts those labels to relative actions and back during inference.
    """

    def __init__(self, image_mappings: dict[str, str]):
        self.image_mappings = dict(image_mappings)

    @staticmethod
    def _pose_vector(value):
        return np.asarray(value, dtype=np.float32).reshape(9)

    @staticmethod
    def _encode_image(frame):
        return image.resize_with_pad_per_frame(*gr00t.IMAGE_SIZE, Image.Resampling.BILINEAR, np.asarray(frame))

    def encode(self, inputs: dict) -> dict:
        return {
            keys.EE_POSE: self._pose_vector(inputs[keys.EE_POSE]),
            keys.GRIP: np.asarray(inputs[keys.GRIP], dtype=np.float32).reshape(1),
            keys.JOINTS: np.asarray(inputs[keys.JOINTS], dtype=np.float32).reshape(7),
            keys.TASK: inputs[keys.TASK],
            **{source: self._encode_image(inputs[source]) for source in self.image_mappings.values()},
        }

    def _decode_single(self, data: dict) -> dict:
        return {
            keys.ROBOT_COMMAND: command.JointPosition(
                positions=np.asarray(data[gr00t.JOINT_POSITION]).reshape(7), mode=DROID_IMPEDANCE
            ),
            keys.TARGET_GRIP: np.asarray(data[gr00t.GRIP]).item(),
        }

    def _derive_pose(self, episode: Episode):
        return tf.Elementwise(episode[keys.EE_POSE], tf.lazy_sequence(self._pose_vector))

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

        def derive_action(episode):
            signals = [derive(episode) for derive in state_encoders.values()]
            timeline = select_timeline(name for signal in signals for name in signal.timelines)
            return tf.concat(*signals, timelines=(timeline,), dtype=np.float32)

        return Derive(
            meta=meta,
            **{
                **state_encoders,
                keys.TASK: Get(keys.TASK, ''),
                ACTION: derive_action,
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
        | ConvertPose(
            geom.Rotation.Representation.ROT6D.value,
            # GR00T's gr00t/data/state_action/droid_frame.py defines this rotation offset.
            rotation_offset=geom.Rotation.from_rotation_matrix(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])),
        )
        | DroidCodec(image_mappings)
        | PackObservationFields(
            {
                gr00t.VIDEO: image_mappings,
                gr00t.STATE: {gr00t.EE_POSE: keys.EE_POSE, gr00t.GRIP: keys.GRIP, gr00t.JOINT_POSITION: keys.JOINTS},
                gr00t.LANGUAGE: {gr00t.TASK: keys.TASK},
            },
            unsqueeze_dims=2,
        )
        | UnpackActionChunk({name: [0, name] for name in (gr00t.JOINT_POSITION, gr00t.GRIP)}, squeeze_dims=1)
    )


droid_three_cameras = droid.override(
    image_mappings={
        gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE,
        gr00t.EXTERIOR_IMAGE_2: keys.EXTERIOR_IMAGE_2,
        gr00t.WRIST_IMAGE: keys.WRIST_IMAGE,
    }
)
