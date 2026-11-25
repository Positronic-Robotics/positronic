"""The replay tail of a RoboLab success: the episode runs ``TAIL_SEC`` more, so that its video shows the result.

RoboLab ends a scene clone at the first step where a success term fires: it freezes the clone and zeroes its action.
A zero action of the DROID jointpos substrate commands every arm joint to 0 rad and opens the gripper, so a frozen
clone does not hold its pose. ``SuccessTail`` delays the success termination instead, and the clone runs on live,
under the commands of its client, until the tail ends.

This module must work in RoboLab's isolated interpreter without Positronic installed.
"""

import math
from typing import NamedTuple

import numpy as np

TAIL_SEC = 1.0

# RoboLab resets a clone in place, and records no verdict, when it terminates within this many control steps: such a
# termination is a physics artifact (``robolab.core.environments.env.RobolabEnv._reset_idx``).
RESET_IN_PLACE_STEPS = 2


class TailStep(NamedTuple):
    """Per scene clone, at one control step: whether RoboLab ends the episode as a success, and whether a success term
    fired at an earlier step of the episode."""

    ended: np.ndarray
    succeeded_earlier: np.ndarray

    def verdict_faults(self, faults: list[str | None]) -> list[str | None]:
        """Per clone, the fault of its arm state at this step, or ``None`` when its success came at an earlier
        step: such a fault only ends the tail."""
        return [None if earlier else fault for fault, earlier in zip(faults, self.succeeded_earlier, strict=True)]


class SuccessTail:
    """Per scene clone, whether a success term fired in this episode, and the control steps since the first one.

    A success stays a success, and the tail does not change the verdict. Its episode ends ``TAIL_SEC`` after the first
    success, or earlier at the time limit or at a fault of the clone's arm state.
    """

    def __init__(self, num_envs: int, control_dt: float):
        self._tail_steps = math.ceil(round(TAIL_SEC / control_dt, 9))
        self._succeeded = np.zeros(num_envs, dtype=bool)
        self._steps_since = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> None:
        self._succeeded[:] = False
        self._steps_since[:] = 0

    def step(
        self, fired: np.ndarray, timed_out: np.ndarray, faulted: np.ndarray, episode_steps: np.ndarray
    ) -> TailStep:
        """Per clone, whether RoboLab ends its episode as a success at this step, and whether it succeeded earlier.

        ``fired`` is whether a success term fires at this step, ``timed_out`` whether the time limit ends the episode
        at this step, ``faulted`` whether the clone's arm state has a fault, and ``episode_steps`` the control steps
        of the episode so far. A success term that fires within ``RESET_IN_PLACE_STEPS`` passes through, so that
        RoboLab resets the clone in place.
        """
        artifact = fired & (episode_steps <= RESET_IN_PLACE_STEPS)
        succeeded_earlier = self._succeeded.copy()
        self._steps_since[self._succeeded] += 1
        self._succeeded |= fired & ~artifact
        tail_over = (self._steps_since >= self._tail_steps) | timed_out | faulted
        return TailStep(ended=artifact | (self._succeeded & tail_over), succeeded_earlier=succeeded_earlier)
