import io
import sys
from functools import partial
from types import SimpleNamespace

import pytest

import pimm
from positronic.eval import Embodiment, Task
from positronic.inference import KeyboardOperator, real
from positronic.tests.testing_coutils import drive_scheduler, scripted_driver


class _IdlePolicy:
    """Enough policy for the attended path to warm up and close; it is never asked for an action."""

    def __init__(self):
        self.warmed = False
        self.closed = False

    def new_session(self, *_args, **_kwargs):
        self.warmed = True
        return SimpleNamespace(close=lambda: None)

    def close(self):
        self.closed = True


def _embodiment(simulated: bool = False) -> Embodiment:
    return Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={},
        static_meta={},
        meta_source=None,
        simulated=simulated,
    )


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: ``KeyboardControl`` returns, the world stops, ``real`` closes the policy.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), task='stub')

    assert policy.closed
    # Not warmed: an attended run opens no throwaway session here. A binary that wants an endpoint
    # driven through its cold start does that itself, before it calls in.
    assert not policy.warmed


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), task='stub')


def test_the_operator_reports_an_ask_the_harness_refuses(capsys):
    """The operator does not police who may start an episode: every press is asked for, and what comes back
    is printed as it lands."""
    task = Task(instruction_source='pick', timeout_sec=None)
    operator = KeyboardOperator(task)
    with pimm.World(virtual_time=True) as world:
        keystrokes = world.pair(operator.keystrokes)
        harness = world.pair(operator.perform_task)
        received = []

        def refuse_a_second_ask():
            """Stand in for the harness's one-episode-at-a-time rule: hold the first call, refuse the rest."""
            for call in harness.incoming():
                received.append(call.request)
                if len(received) > 1:
                    call.set_exception(RuntimeError('An episode is already running'))

        press = partial(keystrokes.emit, 's')
        driver = scripted_driver((press, 0.05), (refuse_a_second_ask, 0.05), (press, 0.05), (refuse_a_second_ask, 0.05))
        drive_scheduler(world.start([operator, driver]))

    assert received == [task, task]
    assert 'already running' in capsys.readouterr().out
