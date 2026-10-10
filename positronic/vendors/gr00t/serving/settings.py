"""GR00T recipe settings shared by dataset preparation, training and serving."""

from typing import Any

from . import EE_POSE, EXTERIOR_IMAGE, EXTERIOR_IMAGE_2, GRIP, JOINT_POSITION, TASK, WRIST_IMAGE
from . import keys as obs_keys

MODEL_SETTINGS = 'model_settings'
ACTION_FPS = 'action_fps'
IMAGE_SIZE = 'image_size'
IMAGE_MAPPINGS = 'image_mappings'
EE_FRAME = 'ee_frame'
ROTATION_OFFSET = 'rotation_offset'
OBSERVATION_KEYS = 'observation_keys'
CONTROL_MODE = 'control_mode'


def droid() -> dict[str, Any]:
    return {
        OBSERVATION_KEYS: {
            EE_POSE: obs_keys.EE_POSE,
            JOINT_POSITION: obs_keys.JOINTS,
            GRIP: obs_keys.GRIP,
            TASK: obs_keys.TASK,
        },
        IMAGE_SIZE: [320, 180],
        IMAGE_MAPPINGS: {EXTERIOR_IMAGE: obs_keys.EXTERIOR_IMAGE, WRIST_IMAGE: obs_keys.WRIST_IMAGE},
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


def droid_three_cameras(exterior_2: str = obs_keys.EXTERIOR_IMAGE_2) -> dict[str, Any]:
    settings = droid()
    cameras = settings[IMAGE_MAPPINGS]
    settings[IMAGE_MAPPINGS] = {
        EXTERIOR_IMAGE: cameras[EXTERIOR_IMAGE],
        EXTERIOR_IMAGE_2: exterior_2,
        WRIST_IMAGE: cameras[WRIST_IMAGE],
    }
    return settings
