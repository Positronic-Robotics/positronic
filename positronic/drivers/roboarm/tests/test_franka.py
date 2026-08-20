from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.roboarm import RobotStatus, command, franka
from positronic.tests.testing_coutils import RecordingEmitter

HOME = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
JOGGED = HOME + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class Call(StrEnum):
    """The vendor calls ``FakeArm`` records."""

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
    set, is what every call but ``stop`` raises; ``error`` is the vendor fault flag every state carries.
    """

    def __init__(self, q, *, polls_to_reach: int = 2, goal_status: 'franka.pf.GoalStatus | None' = None):
        self.q = np.asarray(q, dtype=np.float64)
        self.error = 0
        self.calls: list[Call] = []
        self.targets: list[np.ndarray] = []
        self.raises: Exception | None = None
        self.raises_once: Exception | None = None
        self.polls_to_reach = polls_to_reach
        self._polls = 0
        self.goal_status = goal_status

    def _record(self, call: Call) -> None:
        self.calls.append(call)
        if self.raises_once is not None:
            once, self.raises_once = self.raises_once, None
            raise once
        if self.raises is not None:
            raise self.raises

    def state(self) -> _ArmState:
        self._record(Call.STATE)
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return _ArmState(self.q.copy(), np.zeros(7), pose, np.zeros(6), self.error, '')

    def goal(self) -> _Goal:
        self._record(Call.GOAL)
        self._polls += 1
        if self.goal_status is not None:
            return _Goal(self.goal_status, 'scripted')
        if self._polls >= self.polls_to_reach:
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
    robot._robot = arm  # `_vendor` hands back an already-set handle, which is how the fake arm gets in
    return robot


def _drive(loop, clock: MockClock | None = None) -> None:
    """Pump a driver loop to exhaustion, standing in for the world by advancing ``clock`` through each Sleep."""
    clock = clock or MockClock()
    for sleep in loop:
        clock.advance(sleep.seconds)


def _drive_park(driver: franka.Robot, arm: FakeArm) -> MockClock:
    """Park ``arm`` under a clock that moves only by the waits the park itself asks for."""
    clock = MockClock()
    _drive(driver._arm(StopFlag(), clock).park(), clock)
    return clock


def test_park_drives_the_arm_to_the_home_pose():
    arm = FakeArm(JOGGED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    np.testing.assert_allclose(arm.targets, [HOME])
    np.testing.assert_allclose(arm.q, HOME)


def test_park_commands_the_exact_home_pose_from_inside_the_homing_spread():
    variation = [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
    arm = FakeArm(HOME + np.asarray(variation))

    _drive_park(_driver(arm, variation=variation, manage_desk=False), arm)

    # An arm inside the spread, or travelling through it, is not asked to be judged already home: the
    # controller reports arrival, and a pose it already holds arrives at once.
    np.testing.assert_allclose(arm.targets, [HOME])
    np.testing.assert_allclose(arm.q, HOME)


def test_the_park_waits_by_yielding_rather_than_blocking():
    """A driver's waits are the world's to honour, teardown included: the park asks for them with Sleep."""
    arm = FakeArm(JOGGED, polls_to_reach=3)

    commands = list(_driver(arm, manage_desk=False)._arm(StopFlag(), MockClock()).park())

    assert commands and all(isinstance(command, pimm.Sleep) for command in commands)


def test_park_gives_up_when_the_goal_stops_advancing():
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    assert arm.calls.count(Call.GOAL) == 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_gives_up_when_the_arm_does_not_arrive_in_time():
    arm = FakeArm(JOGGED, polls_to_reach=10**9)

    clock = _drive_park(_driver(arm, manage_desk=False, park_timeout_s=0.05), arm)

    # It waits out the timeout and gives up within one poll interval of it.
    assert 0.05 <= clock.now() < 0.06
    assert arm.calls.count(Call.GOAL) > 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_swallows_a_robot_that_fails_mid_move():
    arm = FakeArm(JOGGED)
    arm.raises = RuntimeError('libfranka: connection lost')

    _drive_park(_driver(arm, manage_desk=False), arm)

    np.testing.assert_allclose(arm.q, JOGGED)


def test_teardown_parks_the_arm_before_stopping_control(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    for _ in range(3):
        next(loop)
    arm.q = JOGGED  # the operator jogs the arm, then finishes the run from there
    mark = len(arm.calls)
    stop.stopped = True
    _drive(loop, clock)

    teardown = arm.calls[mark:]
    assert teardown.index(Call.SET_TARGET_JOINTS) < teardown.index(Call.STOP)
    np.testing.assert_allclose(arm.targets[-1], HOME)
    np.testing.assert_allclose(arm.q, HOME)
    assert desk.prepared and desk.released


def test_teardown_stops_control_and_releases_desk_when_parking_fails(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    for _ in range(3):
        next(loop)
    arm.q = JOGGED
    arm.raises = RuntimeError('libfranka: connection lost')
    mark = len(arm.calls)
    stop.stopped = True
    _drive(loop, clock)

    # the park was attempted, and its failure went no further
    assert arm.calls[mark:] == [Call.RECOVER_FROM_ERRORS, Call.STOP]
    assert desk.released


def test_a_control_fault_stops_the_arm_without_parking_it(desk):
    arm = FakeArm(HOME)
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    for _ in range(3):
        next(loop)
    arm.q = JOGGED
    arm.raises_once = RuntimeError('libfranka: connection lost')  # the fault, not a dead arm — a park could move it
    mark = len(arm.calls)
    with pytest.raises(RuntimeError):
        _drive(loop, clock)

    assert Call.SET_TARGET_JOINTS not in arm.calls[mark:]  # a fault is not answered with autonomous motion
    assert arm.calls[-1] == Call.STOP
    assert desk.released


def test_a_stop_during_the_reset_ends_the_run_without_a_fault(desk):
    """The event that ends the world also cancels the in-flight goal, so a poll taken after the stop
    reports failure. Reading it would turn a clean shutdown into a control fault — which skips the park."""
    arm = FakeArm(JOGGED, polls_to_reach=10**9)  # the opening reset never lands on its own
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    next(loop)  # suspended inside the reset's travel
    stop.stopped = True
    arm.goal_status = franka.pf.GoalStatus.ABORTED

    _drive(loop, clock)

    assert arm.calls[-1] == Call.STOP


def _mover(world: pimm.World, driver: franka.Robot) -> pimm.calls.Caller[command.CommandType, None]:
    """A caller on ``driver.sync_move``, for a test that pumps its generator rather than running a World."""
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w


def test_an_arm_that_will_not_home_reads_error_rather_than_ending_the_run():
    """The driver's own move is the one thing that can fail before a caller exists to hear about it. It reports
    the arm as it is — not where it was put — and keeps running, so the rig is diagnosable rather than gone."""
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)  # the opening homing never lands
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(5):
        next(loop)

    assert states.emitted, 'the driver published nothing'
    assert states.emitted[-1][1].status == RobotStatus.ERROR


def test_a_sync_move_answers_once_the_arm_is_there(world):
    """What a sync move adds over a command: a command is fire and forget, so a caller that must know the arm
    is in place has nothing to wait on. The answer is that arrival."""
    arm = FakeArm(HOME, polls_to_reach=3)  # more than one poll, so an answer cannot land in the asking round
    driver = _driver(arm, manage_desk=False)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening reset
        next(loop)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))
    next(loop)
    assert not answer.done(), 'answered before the arm could have arrived'

    for _ in range(20):
        if answer.done():
            break
        next(loop)

    answer.result()
    np.testing.assert_allclose(arm.targets[-1], JOGGED)
    np.testing.assert_allclose(arm.q, JOGGED)


def test_a_sync_move_the_arm_cannot_make_fails_the_asker(world):
    """A move that stops advancing is the asker's failure to hear about — it is what they were waiting on, and
    silence would hold them for the rest of the run."""
    arm = FakeArm(HOME)
    driver = _driver(arm, manage_desk=False)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):
        next(loop)
    arm.goal_status = franka.pf.GoalStatus.ABORTED
    answer = _mover(world, driver)(command.JointPosition(JOGGED))
    for _ in range(20):
        if answer.done():
            break
        next(loop)

    with pytest.raises(RuntimeError, match='stopped short'):
        answer.result()


def test_an_arm_that_stopped_short_reads_error_until_a_move_lands(world):
    """A move that fails leaves the arm somewhere nobody asked for. The vendor reports its own faults, and a
    stall is not one of them, so without this the next state reads AVAILABLE at a pose no caller chose."""
    arm = FakeArm(HOME)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    move = _mover(world, driver)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening reset
        next(loop)
    arm.goal_status = franka.pf.GoalStatus.ABORTED
    answer = move(command.JointPosition(JOGGED))
    for _ in range(20):
        if answer.done():
            break
        next(loop)
    next(loop)

    assert states.emitted[-1][1].status == RobotStatus.ERROR

    arm.goal_status = None  # whatever stalled the arm is cleared
    answer = move(command.JointPosition(HOME))
    for _ in range(20):
        if answer.done():
            break
        next(loop)
    answer.result()
    next(loop)

    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE


def test_the_state_answering_a_sync_move_carries_the_pose_the_arm_reached(world):
    """The poll that reports arrival is the one the travel stops on, so the sample before it was taken while the
    arm was still moving. Publishing that one hands the caller ``AVAILABLE`` at the pose it set out from."""
    arm = FakeArm(HOME, polls_to_reach=3)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening reset
        next(loop)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))
    for _ in range(20):
        if answer.done():
            break
        next(loop)

    answer.result()
    arrived = states.emitted[-1][1]
    assert arrived.status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(arrived.q, JOGGED)


def test_a_move_that_lands_as_its_deadline_expires_is_an_arrival():
    """The deadline stops the poll loop before it asks again, so a goal that landed just then has not been
    seen. Reading a timeout off the clock alone would fail a caller whose arm is exactly where it asked."""
    arm = FakeArm(HOME, polls_to_reach=10**9)  # it never lands on a poll of its own
    driver = _driver(arm, manage_desk=False)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    travel = driver._arm(StopFlag(), clock).move_to(JOGGED)

    next(travel)  # the first poll: the goal is in flight
    clock.advance(60.0)  # the deadline expires
    arm.goal_status = franka.pf.GoalStatus.REACHED  # and the goal lands in the same moment

    with pytest.raises(StopIteration) as done:
        next(travel)
    assert done.value.value is franka.MoveStatus.ARRIVED
    # Only the goal itself was commanded: an arrival is not answered with a hold at where the arm stands
    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1


def test_a_fault_that_lands_with_the_arrival_reads_error_rather_than_available():
    """A reflex between the goal poll and the state read leaves the arm faulted at the target it reached. The
    state answering the move reports the arm as the vendor describes it, not the arrival the goal reported."""
    arm = FakeArm(HOME, polls_to_reach=1)  # the first poll of the goal already reports it reached
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    travel = driver._arm(StopFlag(), MockClock()).move_to(JOGGED)

    arm.error = 1
    with pytest.raises(StopIteration) as done:
        next(travel)

    assert done.value.value is franka.MoveStatus.ARRIVED
    assert states.emitted[-1][1].status == RobotStatus.ERROR


def test_a_sync_move_that_never_arrives_times_out_and_holds_where_the_arm_stopped(world):
    """A goal the controller never converges on is not an error the vendor reports, so without a deadline the
    asker waits for the rest of the run — and the arm keeps chasing a target it was told it failed."""
    arm = FakeArm(HOME)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    move = _mover(world, driver)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening reset, which still lands
        next(loop)
    arm.polls_to_reach = 10**9  # from here the goal stays in flight
    answer = move(command.JointPosition(JOGGED))
    for _ in range(int(franka._Arm._MOVE_GRACE_S) * 4):
        if answer.done():
            break
        clock.advance(1.0)
        next(loop)

    with pytest.raises(TimeoutError, match='stopped short'):
        answer.result()
    np.testing.assert_allclose(arm.targets[-1], HOME)
    np.testing.assert_allclose(arm.targets[-2], JOGGED)
    # Published before the asker heard: a caller that starts recovering must not read the arm as available
    assert states.emitted[-1][1].status == RobotStatus.ERROR
