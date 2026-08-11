import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

import pimm
from pimm.world import SystemClock
from positronic.drivers.roboarm import franka

HOME = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
JOGGED = HOME + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class Call(StrEnum):
    """The vendor calls ``FakeArm`` records, named once for the fake and its assertions alike."""

    STATE = 'state'
    GOAL = 'goal'
    SET_TARGET_JOINTS = 'set_target_joints'
    RECOVER_FROM_ERRORS = 'recover_from_errors'
    STOP = 'stop'
    SET_COLLISION_BEHAVIOR = 'set_collision_behavior'
    SET_CONTROL_MODE = 'set_control_mode'
    SET_LOAD = 'set_load'


@dataclass
class _Goal:
    status: franka.pf.GoalStatus
    reason: str | None


@dataclass
class _ArmState:
    q: np.ndarray
    dq: np.ndarray
    end_effector_pose: np.ndarray
    ee_wrench: np.ndarray
    error: int
    error_message: str


class FakeArm:
    """In-memory ``pf.Robot``: a commanded joint target is reached after ``polls_to_reach`` reads of ``goal``.

    ``goal_status`` pins the reported status, so a move that never lands can be scripted; ``raises``, once
    set, is what every call but ``stop`` raises.
    """

    def __init__(self, q, *, polls_to_reach: int = 2, goal_status: 'franka.pf.GoalStatus | None' = None):
        self.q = np.asarray(q, dtype=np.float64)
        self.calls: list[Call] = []
        self.targets: list[np.ndarray] = []
        self.raises: Exception | None = None
        self.raises_once: Exception | None = None
        self._polls_to_reach = polls_to_reach
        self._polls = 0
        self._goal_status = goal_status

    def _record(self, call: Call) -> None:
        self.calls.append(call)
        if self.raises_once is not None:
            once, self.raises_once = self.raises_once, None
            raise once
        if self.raises is not None:
            raise self.raises

    def state(self) -> _ArmState:
        self._record(Call.STATE)
        return _ArmState(self.q.copy(), np.zeros(7), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(6), 0, '')

    def goal(self) -> _Goal:
        self._record(Call.GOAL)
        self._polls += 1
        if self._goal_status is not None:
            return _Goal(self._goal_status, 'scripted')
        if self._polls >= self._polls_to_reach:
            self.q = self.targets[-1].copy()
            return _Goal(franka.pf.GoalStatus.REACHED, None)
        return _Goal(franka.pf.GoalStatus.IN_FLIGHT, None)

    def set_target_joints(self, target) -> None:
        self._record(Call.SET_TARGET_JOINTS)
        self.targets.append(np.asarray(target, dtype=np.float64))
        self._polls = 0

    def recover_from_errors(self) -> None:
        self._record(Call.RECOVER_FROM_ERRORS)

    def stop(self) -> None:
        self.calls.append(Call.STOP)

    def get_robot_model(self) -> str:
        return (Path(franka.__file__).parent / 'fr3.urdf').read_text()

    def set_collision_behavior(self, **thresholds) -> None:
        self._record(Call.SET_COLLISION_BEHAVIOR)

    def set_control_mode(self, mode) -> None:
        self._record(Call.SET_CONTROL_MODE)

    def set_load(self, *load) -> None:
        self._record(Call.SET_LOAD)


class FakeDesk:
    """In-memory ``Desk``: records that the session opened the brakes and released control."""

    def __init__(self):
        self.prepared = False
        self.released = False

    def __enter__(self) -> 'FakeDesk':
        return self

    def __exit__(self, *exc_info) -> bool:
        self.released = True
        return False

    def prepare(self) -> None:
        self.prepared = True


class StopFlag(pimm.SignalReceiver[bool]):
    """``should_stop`` under the test's control."""

    def __init__(self):
        self.stopped = False

    def read(self) -> pimm.Message[bool]:
        return pimm.Message(self.stopped)


@pytest.fixture
def desk(monkeypatch) -> FakeDesk:
    monkeypatch.setenv(franka.DESK_USER_ENV, 'user')
    monkeypatch.setenv(franka.DESK_PASSWORD_ENV, 'password')
    session = FakeDesk()
    monkeypatch.setattr(franka, 'Desk', lambda *credentials: session)
    return session


def _driver(arm: FakeArm, *, variation: list[float] | None = None, **kwargs) -> franka.Robot:
    robot = franka.Robot('192.0.2.1', home_joints=list(HOME), home_joints_variation=variation or [0.0] * 7, **kwargs)
    robot._robot = arm  # _ensure_robot hands back an already-set handle, which is how the fake arm gets in
    return robot


def test_park_drives_the_arm_to_the_home_pose():
    arm = FakeArm(JOGGED)

    _driver(arm, manage_desk=False)._park(arm)

    np.testing.assert_allclose(arm.targets, [HOME])
    np.testing.assert_allclose(arm.q, HOME)


def test_park_commands_the_exact_home_pose_from_inside_the_homing_spread():
    variation = [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
    arm = FakeArm(HOME + np.asarray(variation))

    _driver(arm, variation=variation, manage_desk=False)._park(arm)

    # An arm inside the spread, or travelling through it, is not asked to be judged already home: the
    # controller reports arrival, and a pose it already holds arrives at once.
    np.testing.assert_allclose(arm.targets, [HOME])
    np.testing.assert_allclose(arm.q, HOME)


def test_park_gives_up_when_the_goal_stops_advancing():
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)

    _driver(arm, manage_desk=False)._park(arm)

    assert arm.calls.count(Call.GOAL) == 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_gives_up_when_the_arm_does_not_arrive_in_time():
    arm = FakeArm(JOGGED, polls_to_reach=10**9)

    started = time.monotonic()
    _driver(arm, manage_desk=False, park_timeout_s=0.05)._park(arm)
    elapsed = time.monotonic() - started

    assert 0.05 <= elapsed < 2.0
    assert arm.calls.count(Call.GOAL) > 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_swallows_a_robot_that_fails_mid_move():
    arm = FakeArm(JOGGED)
    arm.raises = RuntimeError('libfranka: connection lost')

    _driver(arm, manage_desk=False)._park(arm)

    np.testing.assert_allclose(arm.q, JOGGED)


def test_teardown_parks_the_arm_before_stopping_control(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    loop = _driver(arm).run(stop, SystemClock())

    for _ in range(3):
        next(loop)
    arm.q = JOGGED  # the operator jogs the arm, then finishes the run from there
    mark = len(arm.calls)
    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    teardown = arm.calls[mark:]
    assert teardown.index(Call.SET_TARGET_JOINTS) < teardown.index(Call.STOP)
    np.testing.assert_allclose(arm.targets[-1], HOME)
    np.testing.assert_allclose(arm.q, HOME)
    assert desk.prepared and desk.released


def test_teardown_stops_control_and_releases_desk_when_parking_fails(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    loop = _driver(arm).run(stop, SystemClock())

    for _ in range(3):
        next(loop)
    arm.q = JOGGED
    arm.raises = RuntimeError('libfranka: connection lost')
    mark = len(arm.calls)
    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    # the park was attempted, and its failure went no further
    assert arm.calls[mark:] == [Call.RECOVER_FROM_ERRORS, Call.STOP]
    assert desk.released


def test_a_control_fault_stops_the_arm_without_parking_it(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    loop = _driver(arm).run(stop, SystemClock())

    for _ in range(3):
        next(loop)
    arm.q = JOGGED
    arm.raises_once = RuntimeError('libfranka: connection lost')  # the fault, not a dead arm — a park could move it
    mark = len(arm.calls)
    with pytest.raises(RuntimeError):
        next(loop)

    assert Call.SET_TARGET_JOINTS not in arm.calls[mark:]  # a fault is not answered with autonomous motion
    assert arm.calls[-1] == Call.STOP
    assert desk.released
