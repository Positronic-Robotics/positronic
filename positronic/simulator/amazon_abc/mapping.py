"""Shared task selection and wire conversions for the client and the ABC server.

This module must work in an isolated interpreter without Positronic installed.
"""

import enum
from collections.abc import Sequence
from typing import Any

import numpy as np

# The arms ABC drives, in the order its 14-value action concatenates them.
ARMS = ('left', 'right')
ARM_JOINTS = 6
# The site each arm is measured and driven at: the YAM's ``grasp_site``, ``YAM_GRASP_SITE_FRAME`` from ``GRIPPER_SITE``.
CONTROL_SITE = '{arm}_grasp_site'
JOINT = '{arm}_joint{index}'

SELECT_TASKS = 'tasks'
TASK_NAME = 'name'

TOKEN_TASK = 'task'
TOKEN_SEED = 'seed'
TOKEN_CAMERA_HEIGHT = 'camera_height'
TOKEN_CAMERA_WIDTH = 'camera_width'
TOKEN_CAMERA_VIEW = 'camera_view'


class CameraView(enum.StrEnum):
    """The cameras an ABC scene renders from."""

    ZED_X = 'zed_x'  # a fit to the ZED-X stations of the ABC-130k dataset
    D405 = 'd405'  # ABC's own cameras, which match the RealSense D405 stations of the ABC-130k dataset


META_TASK = 'task'

ABC_OBS_STATE = 'state'  # ``[joints(6), aperture]`` per arm, in the order ABC names its robots.
ABC_OBS_IMAGES = 'images'
ABC_OBS_PROMPT = 'prompt'
ABC_INFO_TASK_EVAL = 'task_eval'
# ABC's verdict on the trial, and, from a task that reports it, the flag that ends the trial early.
ABC_EVAL_EVER_SUCCESS = 'ever_success'
ABC_EVAL_EVER_FAILED = 'ever_failed'
# ABC's own sim eval runs every task for 236 chunks of 15 actions. A maintenance task succeeds only there,
# and a count task is decided there.
ABC_EVAL_STEPS = 236 * 15

# The raw observation this server reports beside the per-arm fields of ``protocol``. Each arm reports its
# end-effector pose in ABC world coordinates, as a wxyz quaternion.
OBS_SIM_STATE = 'sim_state'  # MuJoCo mjSTATE_INTEGRATION vector.
OBS_TASK_EVAL = 'task_eval'  # The task evaluator's numeric metrics, keyed by recording suffix.


def task_names(selection: str | Sequence[str] | None) -> list[str] | None:
    """The tasks a selection names, as a list: it names one task or several, and None selects every task."""
    return None if selection is None else [selection] if isinstance(selection, str) else list(selection)


def task_eval_signals(task_eval: dict[str, Any]) -> dict[str, Any]:
    """The entries of ABC's ``task_eval`` info that record as signals: numeric, non-empty, keyed ``.name``."""
    signals = {}
    for name, value in task_eval.items():
        array = np.asarray(value)
        if array.size and array.dtype.kind in 'biuf':
            # A recorded signal holds a scalar or a 1-D vector per sample.
            signals[f'.{name}'] = array.ravel() if array.ndim else array[()]
    return signals
