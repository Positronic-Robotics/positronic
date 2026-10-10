"""Training recipes for GR00T datasets."""

from functools import partial

import configuronic as cfn
import numpy as np

from positronic import geom, keys
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode, select_timeline
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get, Identity, map_signals
from positronic.policy import training
from positronic.policy.codecs import (
    ACTION,
    GR00T_MODALITY,
    LEROBOT_FEATURES,
    lerobot_action,
    lerobot_image,
    lerobot_vector,
)
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import settings as model_settings
from positronic.vendors.gr00t.serving.settings import MODEL_SETTINGS

settings = cfn.Config(model_settings.droid)


def _derive_pose(source: str, episode: Episode):
    return tf.Elementwise(
        episode[source],
        tf.lazy_sequence(lambda value: np.asarray(value, dtype=np.float32).reshape(gr00t.STATE_DIMS[gr00t.EE_POSE])),
    )


def _derive_grip(source: str, episode: Episode):
    return tf.Elementwise(episode[source], lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1))


@cfn.config(settings=settings)
def droid(settings: dict) -> EpisodeTransform:
    """Prepare recorded absolute trajectories and metadata for GR00T fine-tuning."""
    image_mappings = settings[model_settings.IMAGE_MAPPINGS]
    observation_keys = settings[model_settings.OBSERVATION_KEYS]
    image_size = settings[model_settings.IMAGE_SIZE]
    frame = geom.Transform3D.from_vector(
        np.asarray(settings[model_settings.EE_FRAME]), geom.Rotation.Representation.QUAT
    )
    offset = geom.Transform3D(rotation=geom.Rotation.from_quat(settings[model_settings.ROTATION_OFFSET]))
    state_encoders = {
        gr00t.EE_POSE: partial(_derive_pose, observation_keys[gr00t.EE_POSE]),
        gr00t.GRIP: partial(_derive_grip, observation_keys[gr00t.GRIP]),
        gr00t.JOINT_POSITION: lambda episode: tf.Elementwise(
            episode[observation_keys[gr00t.JOINT_POSITION]], partial(np.asarray, dtype=np.float32)
        ),
    }
    state_meta = {
        name: {gr00t.START: 0, gr00t.END: gr00t.STATE_DIMS[name], gr00t.ORIGINAL_KEY: name} for name in state_encoders
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
            gr00t.VIDEO: {name: {gr00t.ORIGINAL_KEY: name} for name in image_mappings},
            gr00t.ANNOTATION: {gr00t.TASK.removeprefix(gr00t.ANNOTATION + '.'): {gr00t.ORIGINAL_KEY: gr00t.TASK_INDEX}},
        },
        LEROBOT_FEATURES: {
            **{name: lerobot_vector(gr00t.STATE_DIMS[name]) for name in state_encoders},
            **{name: lerobot_image(*image_size) for name in image_mappings},
            ACTION: lerobot_action(start),
        },
    }

    def derive_action(episode):
        signals = [derive(episode) for derive in state_encoders.values()]
        timeline = select_timeline(name for signal in signals for name in signal.timelines)
        return tf.concat(*signals, timelines=(timeline,), dtype=np.float32)

    columns = Derive(
        meta=meta,
        **{
            **state_encoders,
            keys.TASK: Get(observation_keys[gr00t.TASK], ''),
            ACTION: derive_action,
            **{
                name: lambda episode, source=source: image.resize_with_pad(*image_size, signal=episode[source])
                for name, source in image_mappings.items()
            },
        },
    )

    def encode_pose(value):
        vector = np.asarray(value)
        expected = 3 + geom.Rotation.Representation.QUAT.size
        if vector.shape != (expected,):
            raise ValueError(f'Expected a pose vector with {expected} values, got shape {vector.shape}')
        pose = geom.Transform3D.from_vector(vector, geom.Rotation.Representation.QUAT)
        return (pose * offset).as_vector(geom.Rotation.Representation.ROT6D).astype(np.float32)

    return (
        Identity(meta={ACTION_FPS: settings[model_settings.ACTION_FPS], MODEL_SETTINGS: settings})
        | training.ChangeEEFrame(frame, (observation_keys[gr00t.EE_POSE],))
        | map_signals(encode_pose, (observation_keys[gr00t.EE_POSE],))
        | columns
    )


droid_three_cameras = droid.override(settings=cfn.Config(model_settings.droid_three_cameras))
