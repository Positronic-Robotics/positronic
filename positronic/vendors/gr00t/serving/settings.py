"""GR00T recipe settings shared by dataset preparation, training and serving."""

import json
import math
from pathlib import Path
from typing import Any

from . import EE_POSE, EXTERIOR_IMAGE, EXTERIOR_IMAGE_2, GRIP, JOINT_POSITION, TASK, WRIST_IMAGE
from . import keys as obs_keys

MODEL_SETTINGS = 'model_settings'
MODEL_SETTINGS_PATH = Path('meta/positronic_model_settings.json')
ACTION_FPS = 'action_fps'
IMAGE_SIZE = 'image_size'
IMAGE_MAPPINGS = 'image_mappings'
# Where each pose signal the checkpoint reads sits: ``{signal: {SITE: <site of the rig's model>, OFFSET: <vector>}}``,
# the dumped form of ``positronic.drivers.roboarm.models.EEFrame`` keyed by signal.
EE_FRAMES = 'ee_frames'
SITE = 'site'
OFFSET = 'offset'
# ``positronic.drivers.roboarm.models.GRIPPER_SITE``, the site every Franka rig reports its poses at.
GRIPPER_SITE = 'gripper'
ROTATION_OFFSET = 'rotation_offset'
OBSERVATION_KEYS = 'observation_keys'
CONTROL_MODE = 'control_mode'


def validate(settings: dict[str, Any], checkpoint: Path) -> None:
    """Check preprocessing dimensions and agreement with recorded checkpoint conventions."""
    dimensions = settings[IMAGE_SIZE]
    if len(dimensions) != 2 or any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError('image_size must contain positive integer width and height')
    fps = settings[ACTION_FPS]
    if isinstance(fps, bool) or not isinstance(fps, int | float) or not 0 < fps < math.inf:
        raise ValueError('action_fps must be positive and finite')
    recorded_path = checkpoint / MODEL_SETTINGS_PATH
    if recorded_path.exists():
        recorded = json.loads(recorded_path.read_text())
        compared = (IMAGE_SIZE, EE_FRAMES, ROTATION_OFFSET, ACTION_FPS)
        conflicts = [name for name in compared if recorded.get(name) != settings[name]]
        if set(recorded[IMAGE_MAPPINGS]) != set(settings[IMAGE_MAPPINGS]):
            conflicts.append(IMAGE_MAPPINGS)
        if conflicts:
            raise ValueError(f'Launch settings conflict with checkpoint conventions: {sorted(conflicts)}')


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
        EE_FRAMES: {obs_keys.EE_POSE: {SITE: GRIPPER_SITE, OFFSET: [0.0, 0.0, -0.155, 1.0, 0.0, 0.0, 0.0]}},
        ROTATION_OFFSET: [0.5, 0.5, -0.5, -0.5],
        # rules-allow: hardcoded-keys — the wire form of `DROID_IMPEDANCE`; this image cannot import positronic, and
        # test_droid_control_mode_is_the_droid_impedance_on_the_wire pins it to `command.to_wire`.
        CONTROL_MODE: {
            'type': 'impedance',
            'kq': [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
            'kqd': [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
            'kx': [400.0, 400.0, 400.0, 15.0, 15.0, 15.0],
            'kxd': [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
            'frame': [0.0, 0.0, -0.155, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        ACTION_FPS: 15.0,
    }


def droid_three_cameras() -> dict[str, Any]:
    settings = droid()
    cameras = settings[IMAGE_MAPPINGS]
    settings[IMAGE_MAPPINGS] = {
        EXTERIOR_IMAGE: cameras[EXTERIOR_IMAGE],
        EXTERIOR_IMAGE_2: obs_keys.EXTERIOR_IMAGE_2,
        WRIST_IMAGE: cameras[WRIST_IMAGE],
    }
    return settings
