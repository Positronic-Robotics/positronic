import configuronic as cfn

from positronic import keys
from positronic.cfg.embodiment import droid
from positronic.cfg.eval.real.tasks import BATTERIES_TASK, SCISSORS_TASK, SPOONS_TASK, TOWELS_TASK, UNIFIED_TASK
from positronic.cfg.hardware.roboarm import franka_start_pose
from positronic.eval import Eval, Task


@cfn.config(
    embodiment=droid,
    timeout=180,
    trial_count=1,
    setup='Fill the transparent tote with the items to pick, and empty the large grey one.',
)
def _droid_pick_place(embodiment, instruction, timeout, trial_count, setup):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, the task carries the instruction.

    ``setup`` names what a person puts in front of the arm — the rig has no scene of its own to seed. The
    outcome is the operator's annotation, since real has no ground-truth source to compute one from.
    """
    return Eval(
        embodiment,
        [
            Task(
                instruction_source=instruction,
                timeout_sec=timeout,
                prepare_args={keys.ARM: franka_start_pose(), keys.GRIPPER: 0.0, keys.SCENE: setup},
                meta={keys.EVAL_TRIAL_INDEX: trial, keys.EVAL_TRIAL_COUNT: trial_count},
            )
            for trial in range(trial_count)
        ],
    )


pick_place = _droid_pick_place.override(instruction=UNIFIED_TASK)
pick_place_towels = _droid_pick_place.override(instruction=TOWELS_TASK)
pick_place_spoons = _droid_pick_place.override(instruction=SPOONS_TASK)
pick_place_scissors = _droid_pick_place.override(instruction=SCISSORS_TASK)
pick_place_batteries = _droid_pick_place.override(instruction=BATTERIES_TASK)
