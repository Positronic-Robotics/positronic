import configuronic as cfn

from positronic.cfg.embodiment import droid
from positronic.eval import Eval, Task

UNIFIED_TASK = 'Pick all the items one by one from transparent tote and place them into the large grey tote.'
TOWELS_TASK = 'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
SPOONS_TASK = 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.'
SCISSORS_TASK = 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'
BATTERIES_TASK = 'Pick all the batteries one by one from transparent tote and place them into the large grey tote.'


@cfn.config(embodiment=droid, timeout=180)
def _droid_pick_place(embodiment, instruction, timeout):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, the task carries the instruction.

    Real has no scene to seed (``reset=None`` — reset is physical and human) and no privileged ground-truth source
    to record (``privileged={}`` — the droid exposes none), so the outcome is the operator's annotation rather than
    a computed criterion. ``timeout`` is the per-trial wall-clock budget the Harness applies on the unattended path.
    """
    task = Task(instruction=instruction, timeout=timeout)
    return Eval(embodiment, task)


pick_place = _droid_pick_place.override(instruction=UNIFIED_TASK)
pick_place_towels = _droid_pick_place.override(instruction=TOWELS_TASK)
pick_place_spoons = _droid_pick_place.override(instruction=SPOONS_TASK)
pick_place_scissors = _droid_pick_place.override(instruction=SCISSORS_TASK)
pick_place_batteries = _droid_pick_place.override(instruction=BATTERIES_TASK)
