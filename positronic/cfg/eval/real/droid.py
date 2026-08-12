import configuronic as cfn

from positronic.cfg.embodiment import droid
from positronic.cfg.eval.real.tasks import BATTERIES_TASK, SCISSORS_TASK, SPOONS_TASK, TOWELS_TASK, UNIFIED_TASK
from positronic.eval import Eval, Rollout


@cfn.config(embodiment=droid, timeout=180, rollout_count=1)
def _droid_pick_place(embodiment, instruction, timeout, rollout_count):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, each rollout the instruction.

    Real has no scene to seed (no ``reset`` — staging is physical and human) and no privileged ground-truth source
    to record (the droid exposes none), so the outcome is the operator's annotation rather than a computed
    criterion. ``timeout`` is the per-rollout wall-clock budget the Harness applies on the unattended path;
    ``rollout_count`` is how many rollouts it sweeps — real has no seed or task axis, so each is a bare timed
    rollout with an empty scene.
    """
    return [Eval(embodiment, [Rollout(instruction, timeout) for _ in range(rollout_count)])]


pick_place = _droid_pick_place.override(instruction=UNIFIED_TASK)
pick_place_towels = _droid_pick_place.override(instruction=TOWELS_TASK)
pick_place_spoons = _droid_pick_place.override(instruction=SPOONS_TASK)
pick_place_scissors = _droid_pick_place.override(instruction=SCISSORS_TASK)
pick_place_batteries = _droid_pick_place.override(instruction=BATTERIES_TASK)
