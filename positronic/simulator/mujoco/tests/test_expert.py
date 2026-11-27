from enum import Enum

from positronic.policy.keys import HELPER_HAND_BACK, HELPER_TAKE_OVER, POLICY_SOURCE, TELEOPERATOR
from positronic.policy.takeover import ControlObservation
from positronic.simulator.expert import LEAVE_M, Console


class _Phase(Enum):
    FAR = 'far'
    NEAR = 'near'


RANK = {_Phase.FAR: 0, _Phase.NEAR: 1}
POLICY = ControlObservation(source=POLICY_SOURCE, event=None)
EXPERT = ControlObservation(source=TELEOPERATOR, event=None)
COMMANDS = {'motor': 1.0}


def _presses(element: dict) -> tuple[int | None, int | None]:
    return element[HELPER_TAKE_OVER], element[HELPER_HAND_BACK]


def test_an_expert_that_takes_over_at_the_start_presses_take_over_at_its_second_element_and_never_hands_back():
    console = Console(RANK, takes_over_at_start=True)
    first = console.element(POLICY, _Phase.FAR, 1.0, COMMANDS, 10)
    second = console.element(POLICY, _Phase.FAR, 1.0, COMMANDS, 20)
    later = [console.element(EXPERT, phase, 0.0, COMMANDS, 30 + i) for i, phase in enumerate(_Phase)]
    assert _presses(first) == (None, None)
    assert [_presses(element) for element in (second, *later)] == [(20, None)] * 3
    assert second['motor'] == 1.0


def test_an_expert_takes_over_when_the_driver_leaves_its_path_and_hands_back_at_a_phase_of_a_higher_rank():
    console = Console(RANK, takes_over_at_start=False)
    on_path = [console.element(POLICY, _Phase.FAR, distance, COMMANDS, 10 + i) for i, distance in enumerate((1.0, 0.9))]
    left = console.element(POLICY, _Phase.FAR, 0.9 + LEAVE_M + 0.01, COMMANDS, 20)
    same_phase = console.element(EXPERT, _Phase.FAR, 0.5, COMMANDS, 30)
    next_phase = console.element(EXPERT, _Phase.NEAR, 0.5, COMMANDS, 40)
    assert [_presses(element) for element in on_path] == [(None, None)] * 2
    assert [_presses(element) for element in (left, same_phase, next_phase)] == [(20, None), (20, None), (20, 40)]


def test_an_expert_takes_over_when_the_driver_goes_back_to_an_earlier_phase():
    console = Console(RANK, takes_over_at_start=False)
    console.element(POLICY, _Phase.NEAR, 0.1, COMMANDS, 10)
    console.element(POLICY, _Phase.NEAR, 0.1, COMMANDS, 20)
    assert _presses(console.element(POLICY, _Phase.FAR, 0.1, COMMANDS, 30)) == (30, None)
