"""The observation contract of the OpenPI subprocess: one name per field its input transforms pull out.

Which subset a checkpoint reads follows the config it is served under: the ``pi05_positronic`` and
``pi05_libero`` transforms take the concatenated ``STATE`` and two cameras, the ``pi05_droid`` ones take
joints and gripper apart with their own camera names (openpi ``positronic_policy``, ``libero_policy`` and
``droid_policy``). The codec writes the set its config asks for, so the names are spelled once here.
"""

from typing import Any

import numpy as np

STATE = 'observation/state'
IMAGE = 'observation/image'
WRIST_IMAGE = 'observation/wrist_image'

JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
EXTERIOR_IMAGE_LEFT = 'observation/exterior_image_1_left'
WRIST_IMAGE_LEFT = 'observation/wrist_image_left'

PROMPT = 'prompt'


def warm_observation() -> dict[str, Any]:
    """Zero-filled inputs carrying every field above, so one observation warms a checkpoint under any config.

    A config's input transform pulls out the fields it names and ignores the rest, which is what lets one
    observation stand in for the codec that normally feeds the subprocess. Sizes need only stay within the
    model's own bounds: OpenPI pads the state to the action dimension and resizes images itself.
    """
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        STATE: np.zeros(8, dtype=np.float32),
        IMAGE: frame,
        WRIST_IMAGE: frame,
        JOINT_POSITION: np.zeros(7, dtype=np.float32),
        GRIPPER_POSITION: np.zeros(1, dtype=np.float32),
        EXTERIOR_IMAGE_LEFT: frame,
        WRIST_IMAGE_LEFT: frame,
        PROMPT: '',
    }
