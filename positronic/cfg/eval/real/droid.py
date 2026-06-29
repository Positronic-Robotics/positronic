import configuronic as cfn

from positronic.cfg.embodiment import droid
from positronic.cfg.eval import build_trials
from positronic.cfg.eval.real.tasks import BATTERIES_TASK, SCISSORS_TASK, SPOONS_TASK, TOWELS_TASK, UNIFIED_TASK
from positronic.eval import Eval, Task


@cfn.config(embodiment=droid, timeout=180, trial_count=1)
def _droid_pick_place(embodiment, instruction, timeout, trial_count):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, the task carries the instruction.

    Real has no scene to seed (``reset=None`` — reset is physical and human) and no privileged ground-truth source
    to record (``privileged={}`` — the droid exposes none), so the outcome is the operator's annotation rather than
    a computed criterion. ``timeout`` is the per-trial wall-clock budget the Harness applies on the unattended path;
    ``trial_count`` is how many such trials it sweeps (real has no seed or task axis, so each is a bare timed trial).
    """
    task = Task(instruction=instruction, timeout=timeout)
    return Eval(embodiment, task, build_trials(None, trial_count))


pick_place = _droid_pick_place.override(instruction=UNIFIED_TASK)
pick_place_towels = _droid_pick_place.override(instruction=TOWELS_TASK)
pick_place_spoons = _droid_pick_place.override(instruction=SPOONS_TASK)
pick_place_scissors = _droid_pick_place.override(instruction=SCISSORS_TASK)
pick_place_batteries = _droid_pick_place.override(instruction=BATTERIES_TASK)
