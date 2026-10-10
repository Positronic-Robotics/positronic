"""GR00T recipe settings shared by dataset preparation, training and serving."""

from copy import deepcopy
from typing import Any

from . import EE_POSE, EXTERIOR_IMAGE, EXTERIOR_IMAGE_2, GRIP, JOINT_POSITION, TASK, WRIST_IMAGE

MODEL_SETTINGS = 'model_settings'
ACTION_FPS = 'action_fps'
IMAGE_SIZE = 'image_size'
IMAGE_MAPPINGS = 'image_mappings'
EE_FRAME = 'ee_frame'
ROTATION_OFFSET = 'rotation_offset'
OBSERVATION_KEYS = 'observation_keys'
CONTROL_MODE = 'control_mode'


def droid(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = {
        OBSERVATION_KEYS: {EE_POSE: 'robot_state.ee_pose', JOINT_POSITION: 'robot_state.q', GRIP: 'grip', TASK: 'task'},
        IMAGE_SIZE: [320, 180],
        IMAGE_MAPPINGS: {EXTERIOR_IMAGE: 'image.exterior', WRIST_IMAGE: 'image.wrist'},
        EE_FRAME: [0.0, 0.0, -0.085225977, 0.38268343245394154, 0.0, 0.0, 0.9238795324744832],
        ROTATION_OFFSET: [0.5, 0.5, -0.5, -0.5],
        CONTROL_MODE: {
            'type': 'impedance',
            'kq': [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
            'kqd': [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
            'kx': [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
            'kxd': [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
        },
        ACTION_FPS: 15.0,
    }
    if overrides is not None:
        unknown = overrides.keys() - settings.keys()
        if unknown:
            raise ValueError(f'Unknown GR00T settings: {sorted(unknown)}')
        settings.update(overrides)
    return deepcopy(settings)


def droid_three_cameras(
    overrides: dict[str, Any] | None = None, default_exterior_2: str = 'image.exterior_2'
) -> dict[str, Any]:
    settings = droid(overrides)
    cameras = settings[IMAGE_MAPPINGS]
    settings[IMAGE_MAPPINGS] = {
        EXTERIOR_IMAGE: cameras[EXTERIOR_IMAGE],
        EXTERIOR_IMAGE_2: cameras.get(EXTERIOR_IMAGE_2, default_exterior_2),
        WRIST_IMAGE: cameras[WRIST_IMAGE],
    }
    return settings
