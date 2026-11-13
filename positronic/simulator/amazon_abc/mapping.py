"""Shared task selection and wire conversions for the client and the ABC server.

This module must work in an isolated interpreter without Positronic installed.
"""

import enum
import math
from collections.abc import Sequence

# The arms ABC drives, in the order its 14-value action concatenates them.
ARMS = ('left', 'right')
ARM_JOINTS = 6
# The site each arm is measured and driven at: the YAM's ``grasp_site``, ``YAM_GRASP_SITE_FRAME`` from ``GRIPPER_SITE``.
CONTROL_SITE = '{arm}_grasp_site'
# The base frame of the bundled YAM model, in the frame of the base body of an ABC arm: 9.5mm lower.
BUNDLED_YAM_BASE = (0.0, 0.0, -0.0095)
JOINT = '{arm}_joint{index}'
# ABC's camera names.
TOP_CAMERA = 'top'
WRIST_CAMERA = '{arm}'

SELECT_TASKS = 'tasks'
TASK_NAME = 'name'

TOKEN_TASK = 'task'
TOKEN_SEED = 'seed'
TOKEN_CAMERA_HEIGHT = 'camera_height'
TOKEN_CAMERA_WIDTH = 'camera_width'
TOKEN_CAMERA_VIEW = 'camera_view'
TOKEN_EPISODE_STEPS = 'episode_steps'


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
# ABC's own sim eval runs every task for 236 chunks of 15 actions. A maintenance task succeeds only where ABC
# truncates the trial, and a count task is decided there.
ABC_EVAL_STEPS = 236 * 15
ABC_CONTROL_DT = 0.034  # s between two ABC actions


def episode_steps(timeout_sec: float) -> int:
    """The steps ABC runs a trial for, so that the verdict of the last action arrives by ``timeout_sec``."""
    # Rounding drops the float error of a timeout that a whole number of periods makes.
    steps = math.floor(round(timeout_sec / ABC_CONTROL_DT, 9)) - 1
    if steps < 1:
        raise ValueError(f'A timeout of {timeout_sec}s leaves no ABC action before it')
    return steps


# The raw observation this server reports beside the per-arm fields of ``protocol``. Each arm reports its
# end-effector pose in ABC world coordinates, as a wxyz quaternion.
OBS_SIM_STATE = 'sim_state'  # MuJoCo mjSTATE_INTEGRATION vector.
OBS_TASK_EVAL = 'task_eval'  # The task evaluator's numeric metrics, keyed by recording suffix.


def task_names(selection: str | Sequence[str] | None) -> list[str] | None:
    """The tasks a selection names, as a list: it names one task or several, and None selects every task."""
    return None if selection is None else [selection] if isinstance(selection, str) else list(selection)
