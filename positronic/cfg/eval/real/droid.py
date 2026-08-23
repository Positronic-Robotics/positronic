import configuronic as cfn

from positronic import keys
from positronic.cfg.embodiment import droid
from positronic.cfg.eval.real.tasks import BATTERIES_TASK, SCISSORS_TASK, SPOONS_TASK, TOWELS_TASK, UNIFIED_TASK
from positronic.cfg.hardware.roboarm import FRANKA_JOINTS_SPREAD, FRANKA_NOMINAL_JOINTS
from positronic.drivers.roboarm import command
from positronic.eval import Eval, Task


@cfn.config(
    embodiment=droid,
    timeout=180,
    trial_count=1,
    setup='Fill the transparent tote with the items to pick, and empty the large grey one.',
)
def _droid_pick_place(embodiment, instruction, timeout, trial_count, setup):
    """A real droid tote pick-and-place eval: the embodiment is the physical Franka, the task carries the instruction.

    Every trial starts the arm at a pose drawn around the Franka's nominal joints with the fingers open, and asks
    whoever runs it for ``setup`` — the rig has no scene of its own to seed. ``timeout`` is the per-trial
    wall-clock budget the Harness applies; the outcome is the operator's annotation, since real has no
    ground-truth source to compute one from.
    """
    return Eval(
        embodiment,
        [
            Task(
                instruction_source=instruction,
                timeout_sec=timeout,
                prepare_args={
                    keys.ARM: command.sampled_joints(FRANKA_NOMINAL_JOINTS, FRANKA_JOINTS_SPREAD),
                    keys.GRIPPER: 0.0,
                    keys.SCENE: setup,
                },
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
