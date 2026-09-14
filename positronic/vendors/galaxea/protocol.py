"""Full-chunk transport contract for non-commercial G0.5 evaluation.

Imported directly by the isolated Python 3.10 backend; no Positronic imports belong here.
Observations use Galaxea's msgpack/numpy encoding. Responses contain plain lists and scalars.
"""

import numpy as np

PROTOCOL = 'protocol'
FULL_CHUNK_V1 = 'galaxea-full-chunk-v1'
MODEL_ID = 'g05-droid'
ACTIONS = 'actions'
ERROR = 'error'
IMAGES = 'images'
STATE = 'state'
TASK = 'task'
FREQUENCY = 'frequency'
EMBODIMENT_TYPE = 'embodiment_type'
DROID_FRANKA = 'Droid_Franka'
EXTERIOR_IMAGE = 'exterior_image'
WRIST_IMAGE = 'wrist_image'
DUMMY_WRIST_RIGHT = 'dummy_wrist_right'
RIGHT_ARM = 'right_arm'
RIGHT_GRIPPER = 'right_gripper'


def chunk_response(actions: dict[str, np.ndarray]) -> dict:
    """Serialize every predicted step. Each part is an unbatched (time, dimensions) array."""
    arm = actions[RIGHT_ARM]
    if arm.ndim != 2 or arm.shape[0] == 0 or arm.shape[1] != 7:
        raise ValueError(f'Expected a nonempty (T, 7) arm chunk, got {arm.shape}')
    for name, values in actions.items():
        if values.ndim != 2 or values.shape[0] != arm.shape[0] or not np.isfinite(values).all():
            raise ValueError(f'Invalid chunk for {name}: expected {arm.shape[0]} finite steps, got {values.shape}')
    return {ACTIONS: [{name: values[i].tolist() for name, values in actions.items()} for i in range(len(arm))]}
