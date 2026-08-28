import io
import logging
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any

import pytest

import pimm
from positronic import keys
from positronic.drivers import keyboard
from positronic.eval import Embodiment, Task
from positronic.inference import KeyboardOperator, real
from positronic.policy import Policy
from positronic.tests.testing_coutils import IdleSession, drive_scheduler, scripted_driver


class _IdlePolicy(Policy):
    """Enough policy for the attended path to run an episode and close; it commands nothing."""

    def __init__(self):
        self.warmed = False
        self.closed = False
        self.observations: list[dict] = []

    def new_session(self, *_args, **_kwargs):
        self.warmed = True
        return IdleSession(self)

    def close(self):
        self.closed = True


class _ReadyDevices(pimm.ControlSystem):
    """The arm and fingers of a rig that is already wherever it is asked to go."""

    def __init__(self):
        self.arm = pimm.calls.ControlSystemHandler[Any, None](self)
        self.gripper = pimm.calls.ControlSystemHandler[Any, None](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            for handler in (self.arm, self.gripper):
                for call in handler.incoming():
                    call.set_result(None)
            yield pimm.Sleep(0.01)


def _embodiment(simulated: bool = False) -> Embodiment:
    devices = _ReadyDevices()
    return Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={keys.ARM: devices.arm, keys.GRIPPER: devices.gripper},
        static_meta={},
        meta_source=None,
        control_systems=(devices,),
        simulated=simulated,
    )


def _trial(instruction: str = 'stub') -> Callable[[], Task]:
    prepare_args = {keys.ARM: 'start-pose', keys.GRIPPER: 0.0}
    return partial(Task, instruction_source=instruction, timeout_sec=None, prepare_args=prepare_args)


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: the operator returns, the world stops, ``real`` closes the policy.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), next_task=_trial())

    assert policy.closed
    # Not warmed: an attended run opens no throwaway session here. A binary that wants an endpoint
    # driven through its cold start does that itself, before it calls in.
    assert not policy.warmed


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), next_task=_trial())


class _ScriptedKeys:
    """Stands in for a person at the terminal: starts an episode, stops it once it is running, then quits.

    ``policy`` is what says the episode is running. The rig's devices are spawned, so how long they take to
    answer the trial's prepare is nothing a fixed beat could name. The beat between the stop and the quit is
    the window in which the operator prints what the episode ended on.
    """

    def __init__(self, policy, beat_sec: float = 0.3):
        self._policy = policy
        self._beat_sec = beat_sec
        self._started = False
        self._stopped_at: float | None = None

    def __call__(self) -> str | None:
        if not self._started:
            self._started = True
            return 's'
        if self._stopped_at is None:
            if not self._policy.observations:
                return None
            self._stopped_at = time.monotonic()
            return 'p'
        return 'q' if time.monotonic() - self._stopped_at > self._beat_sec else None


@pytest.mark.timeout(30.0)
def test_a_keypress_opens_an_episode_and_another_ends_it(monkeypatch, caplog):
    """The press is the whole start signal: the rig's devices ready, the episode opens on the instruction it
    was given, and it runs until the operator stops it."""
    policy = _IdlePolicy()
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, _ScriptedKeys(policy)))

    with caplog.at_level(logging.INFO):
        real(policy=policy, embodiment=_embodiment(), next_task=_trial('pick up the cube'))

    assert policy.observations, 'the episode never opened'
    assert policy.observations[0][keys.TASK] == 'pick up the cube'
    assert keys.ENDED_BY_OPERATOR in caplog.text
    assert policy.closed


def test_a_press_that_cannot_open_a_session_keeps_the_run(monkeypatch, caplog):
    """A model that refuses a session ends the press, not the run: the operator hears it and the rig stays
    up for the next one."""

    class _RefusingPolicy(Policy):
        def new_session(self, *_args, **_kwargs):
            raise RuntimeError('endpoint down')

    presses = iter(['s'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: Task(instruction_source='pick', timeout_sec=None), _RefusingPolicy())
    with pimm.World(virtual_time=True) as world:
        world.pair(operator.perform_task)
        with caplog.at_level(logging.ERROR):
            drive_scheduler(world.start([operator, scripted_driver((None, 0.05), (None, 0.05))]))

    assert 'endpoint down' in caplog.text


def test_the_operator_declines_a_press_while_an_episode_runs(monkeypatch, caplog):
    """One episode at a time is the operator's own rule: the second press never reaches the harness."""
    task = Task(instruction_source='pick', timeout_sec=None)
    presses = iter(['s', 's'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: task, _IdlePolicy())
    with pimm.World(virtual_time=True) as world:
        harness = world.pair(operator.perform_task)
        received = []

        def hold_the_ask():
            """Stand in for a harness running the episode it was asked for: take the call, answer nothing."""
            received.extend(call.request.task for call in harness.incoming())

        driver = scripted_driver((hold_the_ask, 0.05), (hold_the_ask, 0.05))
        drive_scheduler(world.start([operator, driver]))

    assert received == [task]
    assert 'already running' in caplog.text
