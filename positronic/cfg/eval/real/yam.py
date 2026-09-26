from functools import partial

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg.embodiment import yam_bimanual
from positronic.cfg.hardware.roboarm import YAM_NOMINAL_JOINTS
from positronic.drivers.roboarm import command
from positronic.eval import Eval, Task
from positronic.eval import keys as eval_keys


def _bimanual_trials(instruction: str, timeout: float | None, trial_count: int) -> list[Task]:
    """Each trial moves both arms to the nominal pose with the grippers open. A person sets the scene."""
    start = {
        f'{eval_keys.ARM}.{side}': command.JointPosition(np.asarray(YAM_NOMINAL_JOINTS)) for side in keys.BIMANUAL_ARMS
    }
    return [
        Task(
            instruction_source=instruction,
            timeout_sec=timeout,
            prepare_args=start,
            meta={eval_keys.TRIAL_INDEX: trial, eval_keys.TRIAL_COUNT: trial_count},
        )
        for trial in range(trial_count)
    ]


@cfn.config(embodiment=yam_bimanual, instruction='fold the towel', timeout=60, trial_count=1)
def bimanual(embodiment, instruction: str, timeout: float | None, trial_count: int):
    """A real bimanual YAM eval: the operator sets the scene and annotates the outcome."""
    return Eval(embodiment, partial(_bimanual_trials, instruction, timeout, trial_count))
