import io
import sys
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import pytest

import pimm
from positronic import keys
from positronic.eval import Embodiment
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
        descriptor='stub', observations={}, commands={}, static_meta={}, meta_source=None, simulated=simulated
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


class _StubHarness(pimm.ControlSystem):
    """Mimics the harness's one-episode-at-a-time rule: it holds the first call and refuses the rest."""

    def __init__(self):
        self.perform_task = pimm.calls.ControlSystemHandler[dict[str, Any], dict[str, Any]](self)
        self.received: list[dict[str, Any]] = []

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            for call in self.perform_task.incoming():
                self.received.append(call.request)
                if len(self.received) > 1:
                    call.set_exception(RuntimeError('An episode is already running'))
            yield pimm.Sleep(0.01)


def test_the_operator_reports_an_ask_the_harness_refuses(capsys):
    """The operator does not police who may start an episode: every press is asked for, and what comes back
    is printed as it lands."""
    operator = KeyboardOperator(task='pick')
    harness = _StubHarness()
    with pimm.World(virtual_time=True) as world:
        keystrokes = cast(pimm.SignalEmitter[str], world.pair(operator.keystrokes))
        world.connect(operator.perform_task, harness.perform_task)
        press = partial(keystrokes.emit, 's')
        driver = scripted_driver((press, 0.05), (press, 0.05), (None, 0.05))
        drive_scheduler(world.start([operator, harness, driver]))

    assert harness.received == [{keys.TASK: 'pick'}, {keys.TASK: 'pick'}]
    assert 'already running' in capsys.readouterr().out
