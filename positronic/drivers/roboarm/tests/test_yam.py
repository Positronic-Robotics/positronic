from collections.abc import Callable, Iterator

import numpy as np
import pytest

import pimm
from positronic import keys
from positronic.cfg import embodiment, video_encoder
from positronic.cfg.eval.real import yam as yam_eval
from positronic.cfg.hardware.roboarm import YAM_NOMINAL_JOINTS
from positronic.drivers.roboarm import command, yam
from positronic.eval import Embodiment
from positronic.eval import keys as eval_keys

OPEN, CLOSED = 0.0, 1.0
_GRIP_TOL = 0.05
STATION_GRAVITY_COMP = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0]


@pytest.fixture
def chains(monkeypatch) -> dict[str, yam._FakeYam]:
    """The fake chain each driver opens, keyed by its CAN channel."""
    opened: dict[str, yam._FakeYam] = {}
    monkeypatch.setattr(yam, 'get_yam_robot', lambda channel, **_: opened.setdefault(channel, yam._FakeYam()))
    return opened


def _run_until(loop: Iterator[pimm.Command], done: Callable[[], bool], steps: int = 20_000) -> None:
    for _ in range(steps):
        if done():
            return
        next(loop)
    raise AssertionError('the world did not reach the state the test waits for')


def _grip(rx: pimm.SignalReceiver[float]) -> float | None:
    msg = rx.read()
    return None if msg is None else msg.data


def _a_trial_after_a_closed_trial(rig: Embodiment, prepare_args: dict, sides: list[str]) -> list[float | None]:
    """Close each gripper as a policy does at the end of a trial, then ready the next trial with
    ``prepare_args``. Returns the grip that each arm reads, in the order of ``sides``, once every prepare call
    has an answer.

    ``sides`` holds the suffix of each arm's ``grip`` and ``target_grip`` signals.
    """
    with pimm.World(virtual_time=True) as world:
        close = {s: world.pair(rig.commands[f'{keys.TARGET_GRIP}{s}'].dest) for s in sides}
        grips = {s: world.pair(rig.observations[f'{keys.GRIP}{s}'].source) for s in sides}
        prepare = {name: world.pair(handler) for name, handler in rig.prepare_handlers.items()}
        loop = world.start(list(rig.control_systems))

        for emitter in close.values():
            emitter.emit(CLOSED)
        _run_until(loop, lambda: all((g := _grip(rx)) is not None and g > CLOSED - _GRIP_TOL for rx in grips.values()))

        answers = [prepare[name](arg) for name, arg in prepare_args.items()]
        _run_until(loop, lambda: all(a.done() for a in answers))
        for answer in answers:
            answer.result()
        return [_grip(grips[s]) for s in sides]


def test_a_bimanual_trial_starts_with_both_grippers_open_after_a_trial_that_closed_them(chains):
    rig = yam_eval.bimanual.override(
        embodiment=embodiment.yam_bimanual.override(cameras={}, video_encoder=video_encoder.libx264_veryfast)
    ).instantiate()
    (trial,) = rig.tasks()

    grips = _a_trial_after_a_closed_trial(rig.embodiment, trial.prepare_args, [f'.{s}' for s in keys.BIMANUAL_ARMS])

    assert all(g is not None and g < _GRIP_TOL for g in grips), grips
    assert len(chains) == 2
    for chain in chains.values():
        assert chain.last_command is not None and chain.last_command[6] == pytest.approx(1.0)  # the chain's 1 is open


def test_a_single_arm_trial_starts_with_the_gripper_open_after_a_trial_that_closed_it(chains):
    rig = embodiment.yam.override(video_encoder=video_encoder.libx264_veryfast).instantiate()
    start = {eval_keys.ARM: command.JointPosition(np.asarray(YAM_NOMINAL_JOINTS)), eval_keys.GRIPPER: OPEN}

    (grip,) = _a_trial_after_a_closed_trial(rig, start, [''])

    assert grip is not None and grip < _GRIP_TOL, grip
    (chain,) = chains.values()
    assert chain.last_command is not None and chain.last_command[6] == pytest.approx(1.0)


def _opened_with(**kwargs) -> dict:
    """Start a ``Robot`` built with ``kwargs`` and return what it asked its vendor factory for."""
    seen = {}

    def connect(channel, sim, gravity_comp_factor):
        seen.update(channel=channel, sim=sim, gravity_comp_factor=gravity_comp_factor)
        return yam._FakeYam()

    robot = yam.Robot('can0', connect=connect, **kwargs)
    with pimm.World() as world:
        loop = world.start([robot])
        next(loop)  # the chain is opened before the driver yields for the first time
    return seen


def test_a_station_hands_its_gravity_compensation_to_the_chain():
    """i2rt holds a joint against a gravity model of its own, and a joint that model reads short settles below
    where it is sent. The factors a station measured are no use to it unless the driver passes them on."""
    passed = _opened_with(gravity_comp_factor=STATION_GRAVITY_COMP)['gravity_comp_factor']
    np.testing.assert_array_equal(passed, STATION_GRAVITY_COMP)


def test_a_station_that_measured_none_leaves_the_vendor_its_own():
    """Every YAM shares i2rt's factors until a station measures better ones; naming none has to mean that,
    rather than a vector of ones that would turn the compensation off."""
    assert _opened_with()['gravity_comp_factor'] is None


def _gravity_by_channel(monkeypatch, rig) -> dict:
    """Start a bimanual ``rig`` and return the gravity compensation each CAN channel asked i2rt for."""
    factors = {}

    def get_yam_robot(channel, gravity_comp_factor, **_):
        factors[channel] = gravity_comp_factor
        return yam._FakeYam()

    monkeypatch.setattr(yam, 'get_yam_robot', get_yam_robot)
    with pimm.World() as world:
        loop = world.start(
            list(rig.override(cameras={}, video_encoder=video_encoder.libx264_veryfast).instantiate().control_systems)
        )
        _run_until(loop, lambda: len(factors) == 2)
    return factors


def test_the_yambox_station_hands_its_gravity_compensation_to_both_chains(monkeypatch):
    for passed in _gravity_by_channel(monkeypatch, embodiment.yam_bimanual_yambox).values():
        np.testing.assert_array_equal(passed, STATION_GRAVITY_COMP)


def test_each_bimanual_chain_gets_the_gravity_compensation_of_its_own_arm(monkeypatch):
    left_channel, right_channel = 'can-left', 'can-right'
    left, right = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0], [1.0, 1.2, 1.3, 1.5, 1.0, 1.0]
    rig = embodiment.yam_bimanual_yambox.override(
        left_channel=left_channel,
        right_channel=right_channel,
        gravity_comp_factor={keys.LEFT_ARM: left, keys.RIGHT_ARM: right},
    )
    factors = _gravity_by_channel(monkeypatch, rig)
    np.testing.assert_array_equal(factors[left_channel], left)
    np.testing.assert_array_equal(factors[right_channel], right)
