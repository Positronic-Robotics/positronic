import io
import logging
import sys
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any

import pytest

import pimm
from positronic import keys
from positronic.drivers import keyboard
from positronic.eval import Embodiment, Task
from positronic.eval import keys as eval_keys
from positronic.inference import KeyboardOperator, real
from positronic.policy import Policy
from positronic.policy.base import Step
from positronic.tests.testing_coutils import drive_scheduler, scripted_driver


class _IdlePolicy(Policy):
    """Enough policy for the attended path to run an episode and close; it commands nothing."""

    def __init__(self):
        self.started = False
        self.closed = False
        self.observations: list[dict] = []

    def run(self, runtime):
        self.started = True
        try:
            obs = yield
            while True:
                self.observations.append(obs)
                obs = yield Step({}, runtime.time_ns + 100_000_000)
        finally:
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
        prepare_handlers={eval_keys.ARM: devices.arm, eval_keys.GRIPPER: devices.gripper},
        static_meta={},
        meta_source=None,
        control_systems=(devices,),
        simulated=simulated,
    )


def _trial(instruction: str = 'stub') -> Callable[[], Task]:
    prepare_args = {eval_keys.ARM: 'start-pose', eval_keys.GRIPPER: 0.0}
    return partial(Task, instruction_source=instruction, timeout_sec=None, prepare_args=prepare_args)


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: the operator returns, the world stops, no policy run is started.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), next_task=_trial())

    assert not policy.closed
    assert not policy.started


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), next_task=_trial())


@pytest.mark.timeout(30.0)
def test_a_keypress_opens_an_episode_and_another_ends_it(monkeypatch, caplog):
    """The press is the whole start signal: the rig's devices ready, the episode opens on the instruction it
    was given, and it runs until the operator stops it."""
    policy = _IdlePolicy()

    def presses():
        yield 's'
        while not policy.observations:
            yield None
        yield 'p'
        while 'Episode ended:' not in caplog.text:
            yield None
        yield 'q'

    script = presses()
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(script, None)))

    with caplog.at_level(logging.INFO):
        real(policy=policy, embodiment=_embodiment(), next_task=_trial('pick up the cube'))

    assert policy.observations, 'the episode never opened'
    assert policy.observations[0][keys.TASK] == 'pick up the cube'
    assert eval_keys.ENDED_BY_OPERATOR in caplog.text
    assert policy.closed


def test_a_task_failure_is_reported_without_stopping_the_operator(monkeypatch, caplog):
    presses = iter(['s'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: Task(instruction_source='pick', timeout_sec=None), _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        handler = world.pair(operator.perform_task)

        def refuse():
            for call in handler.incoming():
                call.set_exception(RuntimeError('endpoint down'))

        with caplog.at_level(logging.ERROR):
            drive_scheduler(world.start([operator, scripted_driver((refuse, 0.1), (None, 0.1))]))
    assert 'endpoint down' in caplog.text


def test_the_operator_declines_a_press_while_an_episode_runs(monkeypatch, caplog):
    """One episode at a time is the operator's own rule: the second press never reaches the harness."""
    task = Task(instruction_source='pick', timeout_sec=None)
    presses = iter(['s', 's'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: task, _IdlePolicy(), None)
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
