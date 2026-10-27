"""Training recipes for GR00T datasets."""

from functools import partial

import configuronic as cfn
import numpy as np
from eval_vocabulary.timeline import recording_timeline

from positronic import geom, keys
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode
from positronic.dataset.time import Time
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get, Identity, map_signals
from positronic.drivers.roboarm.models import parse_ee_frames
from positronic.policy import training
from positronic.policy.codecs import (
    ACTION,
    GR00T_MODALITY,
    LEROBOT_FEATURES,
    lerobot_action,
    lerobot_image,
    lerobot_vector,
)
from positronic.policy.codecs.metadata import MODEL_SETTINGS
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import settings as model_settings


def _derive_pose(source: str, episode: Episode):
    return tf.Elementwise(
        episode[source],
        tf.lazy_sequence(lambda value: np.asarray(value, dtype=np.float32).reshape(gr00t.STATE_DIMS[gr00t.EE_POSE])),
    )


def _derive_grip(source: str, episode: Episode):
    return tf.Elementwise(episode[source], lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1))


@cfn.config(settings=model_settings.droid(), action_lead_sec=0.14, binarize_grip=True)
def droid(settings: dict, action_lead_sec: float, binarize_grip: bool) -> EpisodeTransform:
    """Prepare recorded absolute trajectories and metadata for GR00T fine-tuning."""
    image_mappings = settings[model_settings.IMAGE_MAPPINGS]
    conflicts = image_mappings.keys() & (gr00t.STATE_DIMS.keys() | {ACTION, keys.TASK})
    if conflicts:
        raise ValueError(f'Image names overlap training columns: {sorted(conflicts)}')
    observation_keys = settings[model_settings.OBSERVATION_KEYS]
    image_size = settings[model_settings.IMAGE_SIZE]
    ee_frame = parse_ee_frames(settings[model_settings.EE_FRAMES])[observation_keys[gr00t.EE_POSE]]
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
        signals = {name: derive(episode) for name, derive in state_encoders.items()}
        timeline = recording_timeline({name for signal in signals.values() for name in signal.timelines})
        grip = _derive_grip(keys.TARGET_GRIP, episode)
        if binarize_grip:
            grip = tf.Elementwise(grip, lambda values: (np.asarray(values) > 0.5).astype(np.float32))
        signals[gr00t.GRIP] = grip
        # The arm reaches a target about `action_lead_sec` after the command, so the arm action is the measured state
        # that much later. The recorded command is not used: a rig commands poses or joints, not both. The measured
        # grip stays low when the fingers stop on an object, so the gripper action is the command.
        current = tf.concat(*signals.values(), timelines=(timeline,), dtype=np.float32)
        ahead = tf.TimeOffsets(current, Time(**{timeline: round(action_lead_sec * 1e9)}))
        parts = [
            tf.view(current if name == gr00t.GRIP else ahead, slice(bounds[gr00t.START], bounds[gr00t.END]))
            for name, bounds in action_meta.items()
        ]
        return tf.concat(*parts, timelines=(timeline,), dtype=np.float32)

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
        | training.ChangeEEFrame(ee_frame, (observation_keys[gr00t.EE_POSE],))
        | map_signals(encode_pose, (observation_keys[gr00t.EE_POSE],))
        | columns
    )


droid_three_cameras = droid.override(settings=model_settings.droid_three_cameras())
