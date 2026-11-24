"""The replay tail of a RoboLab success: the episode runs ``TAIL_SEC`` more, so that its video shows the result.

RoboLab ends a scene clone at the first step where a success term fires: it freezes the clone and zeroes its action.
A zero action of the DROID jointpos substrate commands every arm joint to 0 rad and opens the gripper, so a frozen
clone does not hold its pose. ``SuccessTail`` delays the success termination instead, and the clone runs on live,
under the commands of its client, until the tail ends.

This module must work in RoboLab's isolated interpreter without Positronic installed.
"""

import math

import numpy as np

TAIL_SEC = 1.0

# RoboLab resets a clone in place, and records no verdict, when it terminates within this many control steps: such a
# termination is a physics artifact (``robolab.core.environments.env.RobolabEnv._reset_idx``).
RESET_IN_PLACE_STEPS = 2


class SuccessTail:
    """Per scene clone, whether a success term fired in this episode, and the control steps since the first one.

    A success stays a success. Its episode ends ``TAIL_SEC`` after the first success, or at the time limit when that
    comes first.
    """

    def __init__(self, num_envs: int, control_dt: float):
        self._tail_steps = math.ceil(round(TAIL_SEC / control_dt, 9))
        self._succeeded = np.zeros(num_envs, dtype=bool)
        self._steps_since = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> None:
        self._succeeded[:] = False
        self._steps_since[:] = 0

    def step(self, fired: np.ndarray, timed_out: np.ndarray, episode_steps: np.ndarray) -> np.ndarray:
        """Per clone, whether RoboLab ends its episode as a success at this step.

        ``fired`` is whether a success term fires at this step, ``timed_out`` whether the time limit ends the episode
        at this step, and ``episode_steps`` the control steps of the episode so far. A success term that fires within
        ``RESET_IN_PLACE_STEPS`` passes through, so that RoboLab resets the clone in place.
        """
        artifact = fired & (episode_steps <= RESET_IN_PLACE_STEPS)
        self._steps_since[self._succeeded] += 1
        self._succeeded |= fired & ~artifact
        return artifact | (self._succeeded & ((self._steps_since >= self._tail_steps) | timed_out))
