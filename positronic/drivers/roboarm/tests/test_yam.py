"""The brakeless YAM arm is laid onto its joint stops before it loses torque — on a normal stop, and on a
crash inside the run loop. Without that, a chain going limp from the ready pose drops ~0.4 m."""

import inspect

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.roboarm.yam import _PARK_JOINTS, YAM_STOW_JOINTS, ParkTuning, Robot, _Chain, _FakeYam
from positronic.tests.testing_coutils import ManualCommandReceiver

_DT = 0.01  # seconds per pump; matches the driver's 100 Hz tick so the ramp behaves as it does on the rig

# Event tags the recording fake logs, so the fake and the assertions agree on one spelling.
_CMD, _ZERO_TORQUE, _CLOSE = 'cmd', 'zero_torque', 'close'


class _RunLoopCrash(RuntimeError):
    """Stands for the vendor dropping the chain mid-run, so a test can raise it from inside the loop."""


class _RecordingYam(_FakeYam):
    """A ``_FakeYam`` that records the order of vendor calls, so a test can prove the arm reached the safe
    pose before it was cut limp. Setting ``raise_next`` makes the following joint command raise once."""

    def __init__(self, alpha: float = 0.3):
        super().__init__(alpha)
        self.events: list[str] = []
        self.raise_next = False
        self.pos_at_zero_torque: np.ndarray | None = None

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if self.raise_next:
            self.raise_next = False
            raise _RunLoopCrash('the vendor dropped the chain mid-run')
        super().command_joint_pos(joint_pos)
        self.events.append(_CMD)

    def zero_torque_mode(self) -> None:
        self.pos_at_zero_torque = self._pos[:6].copy()
        self.events.append(_ZERO_TORQUE)
        super().zero_torque_mode()

    def close(self) -> None:
        self.events.append(_CLOSE)
        super().close()


class _StatusSpy(pimm.SignalEmitter):
    """Records the status of each emitted state. The driver re-emits one mutable state object, so each status
    is snapshotted at emit time."""

    def __init__(self):
        self.statuses: list[RobotStatus] = []

    def emit(self, data, ts: int = -1):
        self.statuses.append(data.status)


class _Sink(pimm.SignalEmitter):
    def emit(self, data, ts: int = -1):
        pass


def _driven(fake: _RecordingYam, stop: StopFlag, clock: MockClock):
    """A YAM ``Robot`` over ``fake``, its emitters bound and its loop ready to pump."""
    driver = Robot(connect=lambda channel, sim: fake)
    status = _StatusSpy()
    driver.state._bind(status)
    driver.grip._bind(_Sink())
    driver.robot_meta._bind(_Sink())
    driver.target_grip._bind(ManualCommandReceiver())
    driver.commands._bind(ManualCommandReceiver())
    return driver, status, driver.run(stop, clock)


def _pump_until(loop, clock: MockClock, cond, *, max_steps: int = 4000) -> None:
    """Pump the loop, advancing the clock ``_DT`` each step, until ``cond`` reads True."""
    for _ in range(max_steps):
        if cond():
            return
        next(loop)
        clock.advance(_DT)
    raise AssertionError('condition not met within the step budget')


def _pump_to_end(loop, clock: MockClock, *, max_steps: int = 4000) -> None:
    """Pump the loop to ``StopIteration``, advancing the clock ``_DT`` each step. A run that ends by raising
    lets that exception propagate."""
    for _ in range(max_steps):
        try:
            next(loop)
        except StopIteration:
            return
        clock.advance(_DT)
    raise AssertionError('the loop did not finish within the step budget')


def _in_run_loop(status: _StatusSpy) -> bool:
    """The startup park has finished and the main loop is running. The park publishes BUSY while it settles
    and one AVAILABLE as it lands; the main loop publishes AVAILABLE every tick, so a run of them marks it."""
    return len(status.statuses) >= 3 and all(s is RobotStatus.AVAILABLE for s in status.statuses[-3:])


def _assert_stowed_then_limp(fake: _RecordingYam) -> None:
    """The arm read the stow pose when torque was cut, no command went out afterwards, and the handle was
    then given back."""
    assert fake.pos_at_zero_torque is not None, 'the chain was never cut limp'
    np.testing.assert_allclose(fake.pos_at_zero_torque, YAM_STOW_JOINTS, atol=ParkTuning().tol)
    cut = fake.events.index(_ZERO_TORQUE)
    assert _CMD not in fake.events[cut + 1 :], 'a joint command went out after the chain was cut limp'
    assert fake.events[-1] == _CLOSE, 'the handle was not given back last'


def test_a_normal_stop_stows_the_arm_before_it_goes_limp():
    fake = _RecordingYam()
    stop, clock = StopFlag(), MockClock()
    _, status, loop = _driven(fake, stop, clock)

    _pump_until(loop, clock, lambda: _in_run_loop(status))  # let the startup park finish and the loop begin

    stop.stopped = True
    _pump_to_end(loop, clock)

    _assert_stowed_then_limp(fake)


def test_a_crash_in_the_run_loop_still_stows_the_arm_before_it_goes_limp():
    fake = _RecordingYam()
    stop, clock = StopFlag(), MockClock()
    _, status, loop = _driven(fake, stop, clock)

    _pump_until(loop, clock, lambda: _in_run_loop(status))
    fake.raise_next = True  # the next in-loop joint command raises, standing for a driver crash

    with pytest.raises(_RunLoopCrash):
        _pump_to_end(loop, clock)

    _assert_stowed_then_limp(fake)


class _SaggingYam(_RecordingYam):
    """A chain whose position servo holds a fixed residual below its reference, as the real one does. The
    park closes that gap by biasing the reference; a residual wider than the bias it is allowed cannot close.
    """

    def __init__(self, sag: float):
        super().__init__()
        self._sag = sag

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        held = np.asarray(joint_pos, dtype=np.float64).copy()
        held[:6] -= self._sag  # the servo settles this far short of what it was asked for
        super().command_joint_pos(held)


def _stow_error(fake: _RecordingYam) -> float:
    assert fake.pos_at_zero_torque is not None, 'the chain was never cut limp'
    return float(np.max(np.abs(fake.pos_at_zero_torque - YAM_STOW_JOINTS)))


def _run_to_stop(fake: _RecordingYam, tuning: ParkTuning) -> None:
    stop, clock = StopFlag(), MockClock()
    driver = Robot(connect=lambda channel, sim: fake, park_tuning=tuning)
    driver.state._bind(_StatusSpy())
    driver.grip._bind(_Sink())
    driver.robot_meta._bind(_Sink())
    driver.target_grip._bind(ManualCommandReceiver())
    driver.commands._bind(ManualCommandReceiver())
    loop = driver.run(stop, clock)
    for _ in range(600):
        next(loop)
        clock.advance(_DT)
    stop.stopped = True
    _pump_to_end(loop, clock, max_steps=8000)


def test_a_bench_whose_servo_sags_further_than_the_default_bias_is_tuned_not_edited():
    """The residual is a property of one bench, so the driver takes it as configuration. A chain sagging
    0.12 rad cannot be stowed by the default 0.05 rad of bias; widening the bias stows it, with no edit to
    the driver."""
    sag = 0.12

    with_default = _SaggingYam(sag)
    _run_to_stop(with_default, ParkTuning())
    assert _stow_error(with_default) > ParkTuning().tol, 'the default bias should not have closed a 0.12 rad sag'

    with_tuning = _SaggingYam(sag)
    tuned = ParkTuning(max_correction=0.2)
    _run_to_stop(with_tuning, tuned)
    assert _stow_error(with_tuning) < tuned.tol, 'a bias wide enough for this bench should reach the stow pose'


def test_park_takes_only_a_grip_and_means_the_ready_pose():
    """``park(grip)`` means the ready pose: ``target`` defaults to it, so only a teardown names its own."""
    target = inspect.signature(_Chain.park).parameters['target']
    np.testing.assert_array_equal(target.default, _PARK_JOINTS)
