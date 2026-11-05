from collections.abc import Callable
from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg.embodiment import yam_bimanual
from positronic.cfg.hardware.roboarm import YAM_NOMINAL_JOINTS
from positronic.drivers.roboarm import command
from positronic.eval import Eval, Task
from positronic.eval import keys as eval_keys


def _bimanual_trial(instruction: str, timeout: float | None, meta: dict[str, Any] | None = None) -> Task:
    """One trial: both arms move to the nominal pose with the grippers open. A person sets the scene."""
    start = {
        **{f'{eval_keys.ARM}.{s}': command.JointPosition(np.asarray(YAM_NOMINAL_JOINTS)) for s in keys.BIMANUAL_ARMS},
        **{f'{eval_keys.GRIPPER}.{s}': 0.0 for s in keys.BIMANUAL_ARMS},
    }
    return Task(instruction_source=instruction, timeout_sec=timeout, prepare_args=start, meta=meta or {})


def _bimanual_trials(instruction: str, timeout: float | None, trial_count: int) -> list[Task]:
    return [
        _bimanual_trial(instruction, timeout, {eval_keys.TRIAL_INDEX: trial, eval_keys.TRIAL_COUNT: trial_count})
        for trial in range(trial_count)
    ]


@cfn.config(instruction='fold the towel', timeout=None)
def attended_trials(instruction: str, timeout: float | None) -> Callable[[], Task]:
    """The trials an attended bimanual YAM run gets, one per press. Without a ``timeout`` the operator ends the
    episode."""
    return partial(_bimanual_trial, instruction, timeout)


@cfn.config(embodiment=yam_bimanual, instruction='fold the towel', timeout=60, trial_count=1)
def bimanual(embodiment, instruction: str, timeout: float | None, trial_count: int):
    """A real bimanual YAM eval: the operator sets the scene and annotates the outcome."""
    return Eval(embodiment, partial(_bimanual_trials, instruction, timeout, trial_count))
