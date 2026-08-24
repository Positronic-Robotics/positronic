import io
import sys
from functools import partial

import pytest

import pimm
from positronic import inference, keys
from positronic.eval import Embodiment, Task
from positronic.inference import KeyboardOperator, real
from positronic.tests.testing_coutils import IdleSession, drive_scheduler, scripted_driver


class _IdlePolicy:
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


def test_the_keyboard_path_refuses_a_start_the_rig_cannot_be_put_at():
    """What an episode starts from belongs to the rig, so swapping the embodiment and keeping the start pose
    is caught before a run begins rather than at the first press."""
    with pytest.raises(ValueError, match="'gripper'"):
        real(policy=_IdlePolicy(), embodiment=_embodiment(), task='stub', start_grip=0.0)


class _ScriptedKeyboard(pimm.ControlSystem):
    """Stands in for ``KeyboardControl``: types each key a beat apart, then returns as ``q`` would.

    The beat is wide enough that the episode a key asks for has opened before the next one lands.
    """

    def __init__(self, *keystrokes: str):
        self._keystrokes = keystrokes
        self.keyboard_inputs = pimm.ControlSystemEmitter[str](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for key in self._keystrokes:
            yield pimm.Sleep(0.3)
            if should_stop.value:
                return
            self.keyboard_inputs.emit(key)
        yield pimm.Sleep(0.3)


@pytest.mark.timeout(30.0)
def test_the_operator_readies_the_scene_an_attended_trial_runs_in(monkeypatch, capsys):
    """The rig a person sets up by hand readies like any device: the trial asks for the scene, the operator
    answers for it, and the episode runs from there until they stop it."""
    monkeypatch.setattr(inference, 'KeyboardControl', lambda quit_key: _ScriptedKeyboard('s', 'p'))
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), task='pick up the cube')

    assert policy.observations, 'the episode never opened'
    assert policy.observations[0][keys.TASK] == 'pick up the cube'
    assert keys.ENDED_BY_OPERATOR in capsys.readouterr().out
    assert policy.closed


def test_each_attended_trial_readies_the_devices_its_rig_was_given():
    """A rig with an arm and fingers names both every press, and the arm's start pose is drawn afresh each time."""
    nominal, spread = [0.0] * 7, [0.1] * 7
    first = inference._attended_task('pick', nominal, spread, start_grip=0.0)
    second = inference._attended_task('pick', nominal, spread, start_grip=0.0)

    assert set(first.prepare_args) == {keys.ARM, keys.GRIPPER, keys.SCENE}
    assert first.prepare_args[keys.GRIPPER] == 0.0
    arms = [t.prepare_args[keys.ARM].positions for t in (first, second)]
    assert all(abs(q) <= 0.1 for q in arms[0]), arms[0]
    assert not (arms[0] == arms[1]).all(), 'every trial draws its own start pose'


def test_an_attended_trial_names_no_device_its_rig_has_not_got():
    """A rig with no arm to place holds where the last episode left it rather than being asked for a pose."""
    task = inference._attended_task('pick', nominal_joints=(), joints_spread=(), start_grip=None)

    assert set(task.prepare_args) == {keys.SCENE}


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
