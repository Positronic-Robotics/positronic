"""Shared task selection and wire conversions for the client and the ABC server.

This module must work in an isolated interpreter without Positronic installed.
"""

from typing import Any

import numpy as np

# The arms ABC drives, in the order its 14-value action concatenates them.
ARMS = ('left', 'right')
ARM_JOINTS = 6
# The site each arm is measured and driven at; it coincides with the YAM driver's ``DEFAULT_FRAME``.
CONTROL_SITE = '{arm}_grasp_site'
JOINT = '{arm}_joint{index}'

SELECT_TASKS = 'tasks'
TASK_NAME = 'name'

TOKEN_TASK = 'task'
TOKEN_SEED = 'seed'
TOKEN_CAMERA_HEIGHT = 'camera_height'
TOKEN_CAMERA_WIDTH = 'camera_width'

META_TASK = 'task'

ABC_OBS_STATE = 'state'  # ``[joints(6), aperture]`` per arm, in the order ABC names its robots.
ABC_OBS_IMAGES = 'images'
ABC_OBS_PROMPT = 'prompt'
ABC_INFO_SUCCESS = 'task_success'
ABC_INFO_TASK_EVAL = 'task_eval'

# The raw observation this server reports. Every name but the physics state is per arm.
OBS_JOINT_POS = 'joint_pos'
OBS_JOINT_VEL = 'joint_vel'
OBS_EEF_POS = 'eef_pos'  # ABC world coordinates, metres.
OBS_EEF_QUAT = 'eef_quat'  # ABC world orientation, wxyz.
OBS_GRIP = 'grip'  # Closure in [0, 1].
OBS_SIM_STATE = 'sim_state'  # MuJoCo mjSTATE_INTEGRATION vector.
OBS_TASK_EVAL = 'task_eval'  # The task evaluator's numeric metrics, keyed by recording suffix.


def task_eval_signals(task_eval: dict[str, Any]) -> dict[str, Any]:
    """The entries of ABC's ``task_eval`` info that record as signals: numeric, non-empty, keyed ``.name``."""
    signals = {}
    for name, value in task_eval.items():
        array = np.asarray(value)
        if array.size and array.dtype.kind in 'biuf':
            # A recorded signal holds a scalar or a 1-D vector per sample.
            signals[f'.{name}'] = array.ravel() if array.ndim else array[()]
    return signals


def invert_grip(value: Any) -> float:
    """Between positronic's closure (1 closed) and i2rt's aperture (1 open)."""
    return 1.0 - float(np.clip(value, 0.0, 1.0))
