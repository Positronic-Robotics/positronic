"""GR00T settings and client descriptions, importable without Positronic."""

import json
from pathlib import Path
from typing import Any

from positronic_model_server import spec
from positronic_model_server.keys import ACTION_FPS, MODEL_SETTINGS
from positronic_model_server.spec import component, parallel, sequence

from . import (
    EE_POSE,
    EXTERIOR_IMAGE,
    EXTERIOR_IMAGE_2,
    GRIP,
    JOINT_POSITION,
    LANGUAGE,
    STATE,
    STATE_DIMS,
    TASK,
    VIDEO,
    WRIST_IMAGE,
)

DEFAULT_SETTINGS = Path(__file__).with_name('droid.json')
IMAGE_SIZE = 'image_size'
IMAGE_MAPPINGS = 'image_mappings'
EE_FRAME = 'ee_frame'
ROTATION_OFFSET = 'rotation_offset'
OBSERVATION_KEYS = 'observation_keys'
CONTROL_MODE = 'control_mode'


def load_settings(path: Path = DEFAULT_SETTINGS, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = json.loads(Path(path).read_text())
    if overrides is not None:
        unknown = overrides.keys() - settings.keys()
        if unknown:
            raise ValueError(f'Unknown GR00T settings: {sorted(unknown)}')
        settings.update(overrides)
    return json.loads(json.dumps(settings, allow_nan=False))


def three_camera_settings(
    path: Path = DEFAULT_SETTINGS, default_exterior_2: str = 'image.exterior_2'
) -> dict[str, Any]:
    settings = load_settings(path)
    cameras = settings[IMAGE_MAPPINGS]
    settings[IMAGE_MAPPINGS] = {
        EXTERIOR_IMAGE: cameras[EXTERIOR_IMAGE],
        EXTERIOR_IMAGE_2: cameras.get(EXTERIOR_IMAGE_2, default_exterior_2),
        WRIST_IMAGE: cameras[WRIST_IMAGE],
    }
    return settings


def inference(settings: dict[str, Any]) -> dict[str, Any]:
    """Describe the client codec using the checkpoint's image and pose conventions."""
    image_mappings = settings[IMAGE_MAPPINGS]
    observation_keys = settings[OBSERVATION_KEYS]
    state = {observation_keys[name]: {observation_keys[name]: size} for name, size in STATE_DIMS.items()}
    return sequence(
        component(spec.METADATA, values={ACTION_FPS: settings[ACTION_FPS], MODEL_SETTINGS: settings}),
        component(spec.BINARIZE_GRIP_INFERENCE),
        component(spec.CHANGE_EE_FRAME, transform=settings[EE_FRAME], keys=[observation_keys[EE_POSE]]),
        component(
            spec.CONVERT_POSE,
            output_rotation='rot6d',
            rotation_offset=settings[ROTATION_OFFSET],
            keys=[observation_keys[EE_POSE]],
        ),
        parallel(
            component(
                spec.OBSERVATION_CODEC,
                state=state,
                images={source: [source, settings[IMAGE_SIZE]] for source in image_mappings.values()},
                task_field=observation_keys[TASK],
                task_source=observation_keys[TASK],
            ),
            sequence(
                component(spec.SET_CONTROL_MODE, mode=settings[CONTROL_MODE]),
                component(
                    spec.JOINT_POSITION_ACTION,
                    joints_key=JOINT_POSITION,
                    grip_key=GRIP,
                    num_joints=STATE_DIMS[JOINT_POSITION],
                ),
            ),
        ),
        component(
            spec.PACK_OBSERVATION_FIELDS,
            layout={
                VIDEO: image_mappings,
                STATE: {name: observation_keys[name] for name in STATE_DIMS},
                LANGUAGE: {TASK: observation_keys[TASK]},
            },
            unsqueeze_dims=2,
        ),
        component(
            spec.UNPACK_ACTION_CHUNK, fields={name: [0, name] for name in (JOINT_POSITION, GRIP)}, squeeze_dims=1
        ),
    )
