from collections.abc import Callable
from functools import partial

import configuronic as cfn
import numpy as np

from positronic import keys
from positronic.cfg.embodiment import trossen
from positronic.cfg.hardware.roboarm import TROSSEN_NOMINAL_JOINTS
from positronic.drivers.roboarm import command
from positronic.eval import Eval, Task

PARTS_TASK = 'Pick the parts one by one from the table and put them into the transparent box.'


def _trossen_trial(instruction: str, timeout: float | None) -> Task:
    """One trial on the Trossen station. A person lays the parts out before it, so the trial asks for no scene.

    The arm opens every trial at the pose the operator's demonstrations open at: it rests on the lower limit
    of two joints, where half the directions out of it have no solution.
    """
    return Task(
        instruction_source=instruction,
        timeout_sec=timeout,
        prepare_args={keys.ARM: command.JointPosition(np.asarray(TROSSEN_NOMINAL_JOINTS, dtype=np.float64))},
    )


@cfn.config(instruction=PARTS_TASK, timeout=None)
def attended_trials(instruction: str, timeout: float | None) -> Callable[[], Task]:
    """The trials an attended Trossen run gets, one per press. Without a ``timeout`` the operator ends the
    episode."""
    return partial(_trossen_trial, instruction, timeout)


def _planned_trials(instruction: str, timeout: float | None, trial_count: int) -> list[Task]:
    return [_trossen_trial(instruction, timeout) for _ in range(trial_count)]


@cfn.config(embodiment=trossen, instruction=PARTS_TASK, timeout=120, trial_count=1)
def pick_place(embodiment, instruction: str, timeout: float | None, trial_count: int) -> Eval:
    """A real Trossen pick-and-place eval. The outcome is the operator's annotation: a real rig has no
    ground truth to compute one from."""
    return Eval(embodiment, partial(_planned_trials, instruction, timeout, trial_count))
