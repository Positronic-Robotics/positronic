from collections.abc import Callable
from functools import partial
from typing import Any

import configuronic as cfn

from positronic.cfg.embodiment import droid
from positronic.cfg.eval.real.tasks import BATTERIES_TASK, SCISSORS_TASK, SPOONS_TASK, TOWELS_TASK, UNIFIED_TASK
from positronic.cfg.hardware.roboarm import droid_start_pose
from positronic.eval import ARM, EVAL_TRIAL_COUNT, EVAL_TRIAL_INDEX, GRIPPER, Eval, Task


def _droid_trial(instruction: str, timeout: float | None, meta: dict[str, Any] | None = None) -> Task:
    """One droid trial. A person fills the tote before the trial, so the trial asks for no scene."""
    return Task(
        instruction_source=instruction,
        timeout_sec=timeout,
        prepare_args={ARM: droid_start_pose(), GRIPPER: 0.0},
        meta=meta or {},
    )


@cfn.config(instruction=UNIFIED_TASK, timeout=None)
def attended_trials(instruction: str, timeout: float | None) -> Callable[[], Task]:
    """The trials an attended droid run gets, one per press. Without a ``timeout`` the operator ends
    the episode."""
    return partial(_droid_trial, instruction, timeout)


def _planned_trials(instruction: str, timeout: float | None, trial_count: int) -> list[Task]:
    return [
        _droid_trial(instruction, timeout, {EVAL_TRIAL_INDEX: trial, EVAL_TRIAL_COUNT: trial_count})
        for trial in range(trial_count)
    ]


@cfn.config(embodiment=droid, timeout=180, trial_count=1)
def _droid_pick_place(embodiment, instruction, timeout, trial_count):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, the task carries the instruction.

    The outcome is the operator's annotation, since real has no ground-truth source to compute one from.
    """
    return Eval(embodiment, partial(_planned_trials, instruction, timeout, trial_count))


pick_place = _droid_pick_place.override(instruction=UNIFIED_TASK)
pick_place_towels = _droid_pick_place.override(instruction=TOWELS_TASK)
pick_place_spoons = _droid_pick_place.override(instruction=SPOONS_TASK)
pick_place_scissors = _droid_pick_place.override(instruction=SCISSORS_TASK)
pick_place_batteries = _droid_pick_place.override(instruction=BATTERIES_TASK)
