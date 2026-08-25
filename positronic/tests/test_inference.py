import io
import sys
from dataclasses import replace
from functools import partial
from typing import Any

import pytest

import pimm
from positronic import inference, keys
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

    @property
    def meta(self):
        return {}

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


def test_the_keyboard_path_refuses_a_rig_it_cannot_put_at_a_start_pose():
    """Every attended episode places the arm and opens the fingers, so a rig that readies neither is caught
    before a run begins rather than at the first press."""
    bare = replace(_embodiment(), prepare_handlers={}, control_systems=())
    with pytest.raises(ValueError, match="'gripper'"):
        real(policy=_IdlePolicy(), embodiment=bare, task='stub')


class _ScriptedKeyboard(pimm.ControlSystem):
    """Stands in for ``KeyboardControl``: starts an episode, stops it once it is running, returns as ``q`` would.

    ``policy`` is what says the episode is running. The rig's devices are spawned, so how long they take to
    answer the trial's prepare is nothing a beat between keystrokes could name.
    """

    def __init__(self, policy):
        self._policy = policy
        self.keyboard_inputs = pimm.ControlSystemEmitter[str](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        yield pimm.Sleep(0.1)
        self.keyboard_inputs.emit('s')
        while not self._policy.observations:
            if should_stop.value:
                return
            yield pimm.Sleep(0.01)
        self.keyboard_inputs.emit('p')
        yield pimm.Sleep(0.3)


@pytest.mark.timeout(30.0)
def test_a_keypress_opens_an_episode_and_another_ends_it(monkeypatch, capsys):
    """The press is the whole start signal: the rig's devices ready, the episode opens on the instruction it
    was given, and it runs until the operator stops it."""
    policy = _IdlePolicy()
    monkeypatch.setattr(inference, 'KeyboardControl', lambda quit_key: _ScriptedKeyboard(policy))

    real(policy=policy, embodiment=_embodiment(), task='pick up the cube')

    assert policy.observations, 'the episode never opened'
    assert policy.observations[0][keys.TASK] == 'pick up the cube'
    assert keys.ENDED_BY_OPERATOR in capsys.readouterr().out
    assert policy.closed


def test_the_operator_reports_an_ask_the_harness_refuses(capsys):
    """The operator does not police who may start an episode: every press is asked for, and what comes back
    is printed as it lands."""
    task = Task(instruction_source='pick', timeout_sec=None)
    operator = KeyboardOperator(lambda: task)
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
