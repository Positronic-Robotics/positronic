"""GR00T recipe settings shared by dataset preparation, training and serving."""

import json
from pathlib import Path
from typing import Any

from . import EXTERIOR_IMAGE, EXTERIOR_IMAGE_2, WRIST_IMAGE

MODEL_SETTINGS = 'model_settings'
ACTION_FPS = 'action_fps'
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
