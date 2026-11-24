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
        ended.append(tail.step(np.array(row), steps >= time_limit, steps).tolist())
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
    tail.step(np.array([True]), np.array([False]), np.array([3]))
    tail.reset()
    ended = [tail.step(np.array([False]), np.array([False]), np.array([s])) for s in range(4, 40)]
    assert not any(e.any() for e in ended)
