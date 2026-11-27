"""The replay tail of a RoboLab success, stepped as the env's termination manager steps it."""

import numpy as np

from positronic.simulator.robolab import success_tail

_CONTROL_DT = 1 / 15
_TAIL_STEPS = 15  # 1 s at 15 Hz


def _run(fired: list[list[bool]], time_limit: int = 1000, start: int = 3) -> list[list[bool]]:
    """Per control step, from ``start`` on, whether each clone ends as a success; ``fired`` lists each step's terms."""
    tail = success_tail.SuccessTail(len(fired[0]), _CONTROL_DT)
    ended = []
    for step, row in enumerate(fired, start=start):
        steps = np.full(len(row), step)
        ended.append(tail.step(np.array(row), steps >= time_limit, np.zeros(len(row), dtype=bool), steps).tolist())
    return ended


def test_a_success_ends_its_episode_one_second_after_the_first_step_it_fires():
    ended = _run([[True]] * 40)
    assert [i for i, (e,) in enumerate(ended) if e][0] == _TAIL_STEPS


def test_a_success_stays_after_its_term_stops_firing():
    ended = _run([[True]] + [[False]] * 40)
    assert ended[_TAIL_STEPS] == [True]
    assert not any(e for (e,) in ended[:_TAIL_STEPS])


def test_the_time_limit_ends_the_tail_early_as_a_success():
    ended = _run([[True]] * 10, time_limit=8)
    assert [e for (e,) in ended] == [False] * 5 + [True] * 5


def test_a_time_out_with_no_success_is_no_success():
    assert _run([[False]] * 10, time_limit=8) == [[False]] * 10


def test_a_success_term_within_two_steps_passes_through_for_the_reset_in_place():
    assert _run([[True], [False]], start=2) == [[True], [False]]


def test_a_success_term_at_the_third_step_waits_for_its_tail():
    assert _run([[True], [False]], start=3) == [[False], [False]]


def test_each_clone_runs_its_own_tail():
    ended = _run([[True, False]] * 5 + [[True, True]] * 30)
    assert ended[_TAIL_STEPS] == [True, False]
    assert ended[5 + _TAIL_STEPS] == [True, True]


def test_a_reset_clears_the_success_of_the_last_episode():
    tail = success_tail.SuccessTail(1, _CONTROL_DT)
    tail.step(np.array([True]), np.array([False]), np.array([False]), np.array([3]))
    tail.reset()
    ended = [tail.step(np.array([False]), np.array([False]), np.array([False]), np.array([s])) for s in range(4, 40)]
    assert not any(e.any() for e in ended)


def test_a_reset_of_one_clone_keeps_the_tail_of_the_other():
    tail = success_tail.SuccessTail(2, _CONTROL_DT)
    tail.step(np.array([True, True]), np.array([False, False]), np.array([False, False]), np.array([3, 3]))
    tail.reset([0])
    ended = [
        tail.step(np.array([False, False]), np.array([False, False]), np.array([False, False]), np.array([s, s]))
        for s in range(4, 4 + _TAIL_STEPS)
    ]
    assert tail.succeeded().tolist() == [False, True]
    assert not any(e[0] for e in ended) and ended[-1][1]


_FAULT = 'arm out of range'


def _run_faulted(fired: list[bool], faulted: list[bool]) -> tuple[list[bool], list[str | None]]:
    """Per control step of one clone, whether RoboLab ends it as a success, and the fault that decides the verdict."""
    tail = success_tail.SuccessTail(1, _CONTROL_DT)
    ended, faults = [], []
    for step, (fire, fault) in enumerate(zip(fired, faulted, strict=True), start=3):
        earlier = tail.succeeded()
        ended.append(bool(tail.step(np.array([fire]), np.array([False]), np.array([fault]), np.array([step]))[0]))
        faults.append(success_tail.verdict_faults([_FAULT if fault else None], earlier)[0])
    return ended, faults


def test_an_arm_fault_inside_the_tail_ends_it_and_the_success_stays():
    # RoboLab's result and its recorder both read the success of an ended clone from the buffer that ``step`` returns.
    ended, faults = _run_faulted([True] + [False] * 9, [False] * 5 + [True] * 5)
    assert ended == [False] * 5 + [True] * 5
    assert faults == [None] * 10


def test_an_arm_fault_with_no_earlier_success_decides_the_verdict():
    ended, faults = _run_faulted([False] * 5, [False, False, True, True, True])
    assert ended == [False] * 5
    assert faults == [None, None, _FAULT, _FAULT, _FAULT]


def test_an_arm_fault_at_the_step_of_the_first_success_decides_the_verdict():
    ended, faults = _run_faulted([False, True], [False, True])
    assert ended == [False, True]
    assert faults == [None, _FAULT]
