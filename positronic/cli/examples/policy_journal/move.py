"""The policy and the inference that both journal examples use.

``record.py`` and ``replay.py`` import ``Move`` from this module. A journal names the class of its policy,
and a replay refuses a policy of another class, so the two programs must not define it themselves.
"""

from collections.abc import Callable, Sequence

from positronic.policy import Policy, PolicyRun, Runtime
from positronic.policy.base import Commands, Obs
from positronic.policy.journal import Activity
from positronic.policy.processors import ChunkedSchedule

MOTOR = 'motor'
POSITION = 'position'


class Move(Policy):
    """Play the velocity chunks that ``infer`` returns, at 10 Hz. ``version`` names the inference."""

    def __init__(self, infer: Callable[[Obs], Sequence[Commands]], version: int = 1) -> None:
        self._infer = Activity('step_plan', version, infer)

    def run(self, runtime: Runtime) -> PolicyRun:
        return ChunkedSchedule(fps=10).run(runtime, self._infer)


def infer(obs: Obs) -> list[dict[str, int]]:
    return [{MOTOR: 1}, {MOTOR: 2}]
