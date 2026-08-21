from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, command, franka
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import MoveAbandoned, MoveStatus
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

HOME = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
JOGGED = HOME + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
IMPEDANCE = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)


class Call(StrEnum):
    """The vendor calls ``FakeArm`` records."""

    STATE = 'state'
    GOAL = 'goal'
    SET_TARGET_JOINTS = 'set_target_joints'
    RECOVER_FROM_ERRORS = 'recover_from_errors'
    STOP = 'stop'
    SET_COLLISION_BEHAVIOR = 'set_collision_behavior'
    SET_CONTROL_MODE = 'set_control_mode'
    INVERSE_KINEMATICS = 'inverse_kinematics'
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

    ``goal_status`` pins the reported status, so a move that never lands can be scripted; ``aborts`` is how
    many polls report a reflex before the arm behaves; ``recovers`` is whether its error clears; ``raises``,
    once set, is what every call but ``stop`` raises, and ``ik_raises`` what only the solver raises; ``error``
    is the vendor fault flag every state carries.
    """

    # What libfranka calls a collision reflex, as it reaches the driver in an aborted goal's reason
    REFLEX = 'libfranka: Move command aborted: motion aborted by reflex! ["cartesian_reflex"]'

    def __init__(
        self, q, *, polls_to_reach: int = 2, goal_status: 'franka.pf.GoalStatus | None' = None, aborts: int = 0
    ):
        self.q = np.asarray(q, dtype=np.float64)
        self.error = 0
        self.calls: list[Call] = []
        self.targets: list[np.ndarray] = []
        self.modes: list[Any] = []
        self.raises: Exception | None = None
        self.raises_once: Exception | None = None
        self.ik_raises: Exception | None = None
        self.polls_to_reach = polls_to_reach
        self._polls = 0
        self.goal_status = goal_status
        self.aborts = aborts
        self.recovers = True

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
        if self.aborts:
            self.aborts -= 1
            self.error = 1
            return _Goal(franka.pf.GoalStatus.ABORTED, self.REFLEX)
        if self._polls >= self.polls_to_reach:
            self.q = self.targets[-1].copy()
            return _Goal(franka.pf.GoalStatus.REACHED, None)
        return _Goal(franka.pf.GoalStatus.IN_FLIGHT, None)

    def set_target_joints(self, target) -> None:
        self._record(Call.SET_TARGET_JOINTS)
        self.targets.append(np.asarray(target, dtype=np.float64))
        self._polls = 0

    def recover_from_errors(self) -> bool:
        self._record(Call.RECOVER_FROM_ERRORS)
        if self.recovers:
            self.error = 0
        return self.recovers

    def stop(self) -> None:
        self.calls.append(Call.STOP)

    def get_robot_model(self) -> str:
        return (Path(franka.__file__).parent / 'fr3.urdf').read_text()

    def set_collision_behavior(self, **thresholds) -> None:
        self._record(Call.SET_COLLISION_BEHAVIOR)

    def inverse_kinematics_with_limits(self, pose) -> np.ndarray:
        self._record(Call.INVERSE_KINEMATICS)
        if self.ik_raises is not None:
            raise self.ik_raises
        return self.q.copy()

    def set_control_mode(self, mode) -> None:
        self._record(Call.SET_CONTROL_MODE)
        self.modes.append(mode)

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


def _drive(loop, clock: MockClock | None = None) -> MoveStatus | None:
    """Pump a driver loop to exhaustion, standing in for the world by advancing ``clock`` through each Sleep,
    and hand back what it returned."""
    clock = clock or MockClock()
    while True:
        try:
            wait = next(loop)
        except StopIteration as end:
            return end.value
        if isinstance(wait, pimm.Sleep):
            clock.advance(wait.seconds)


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

    assert arm.calls.count(Call.GOAL) == franka._Arm._MOVE_ATTEMPTS  # one poll each, and no waiting between
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


def test_a_command_the_arm_cannot_reach_leaves_the_running_law_alone(desk):
    """A rejected command must not half-apply: the arm would hold its old target under new dynamics."""
    arm = FakeArm(HOME)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)
    mark = len(arm.modes)
    arm.ik_raises = ValueError('out of reach')
    feed.push(command.CartesianPosition(pose=geom.Transform3D.identity, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)

    assert arm.modes[mark:] == [], 'the arm changed law for a command it never executed'


def test_the_law_changes_only_where_the_target_is_published(desk):
    """A switch with anything between it and the target can leave the arm holding its last one under it."""
    arm = FakeArm(HOME)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)  # init + opening reset
    mark = len(arm.calls)
    feed.push(command.JointPosition(positions=JOGGED, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)
    feed.push(command.Reset())
    for _ in range(6):
        next(loop)

    switches = [i for i, c in enumerate(arm.calls[mark:], start=mark) if c is Call.SET_CONTROL_MODE]
    assert switches, 'the commands applied no mode at all'
    assert all(arm.calls[i + 1] is Call.SET_TARGET_JOINTS for i in switches), arm.calls[mark:]


def test_a_joint_target_the_vendor_would_refuse_leaves_the_running_law_alone(desk):
    """A joint command is passed straight through, so what the vendor rejects has to be caught here."""
    arm = FakeArm(HOME)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)
    mark = len(arm.modes)
    feed.push(command.JointPosition(positions=np.full(7, np.nan), mode=IMPEDANCE))
    for _ in range(2):
        next(loop)

    assert arm.modes[mark:] == [], 'the arm changed law for a target it never held'
    assert not any(np.isnan(t).any() for t in arm.targets), 'a NaN target reached the arm'


def test_park_puts_the_arm_under_its_native_law():
    """The home pose is far off, and only the native law shapes the reference on the way there."""
    arm = FakeArm(JOGGED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    assert isinstance(arm.modes[0], franka.pf.InternalImpedance)


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


def test_an_arm_that_will_not_home_reads_error_rather_than_ending_the_run():
    """The driver's own move is the one that can fail before a caller exists to hear about it, so the run
    goes on and the arm reads as it is."""
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


def test_a_reflex_during_the_opening_home_is_cleared_and_the_arm_sent_again(desk):
    """A reflex aborts the goal and leaves the arm in error, so the home it cut short has to be commanded
    again to happen at all."""
    arm = FakeArm(JOGGED, aborts=1)
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    for _ in range(4):
        next(loop)

    np.testing.assert_allclose(arm.q, HOME)
    commanded = [i for i, call in enumerate(arm.calls) if call is Call.SET_TARGET_JOINTS]
    assert len(commanded) == 2, 'the home was commanded once and never again'
    assert Call.RECOVER_FROM_ERRORS in arm.calls[commanded[0] : commanded[1]]


def test_a_move_that_keeps_aborting_gives_up_rather_than_driving_at_it_again():
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)
    clock = MockClock()

    with pytest.raises(RuntimeError, match='on all 3 attempts'):
        _drive(_driver(arm, manage_desk=False)._arm(StopFlag(), clock).move_to(HOME, None), clock)

    assert arm.calls.count(Call.SET_TARGET_JOINTS) == franka._Arm._MOVE_ATTEMPTS


def test_a_move_gives_up_at_once_on_an_error_the_arm_will_not_clear():
    """An arm that cannot be cleared cannot be commanded either, so the attempts left are worth nothing."""
    arm = FakeArm(JOGGED, aborts=1)
    arm.recovers = False
    clock = MockClock()

    with pytest.raises(RuntimeError, match='will not clear'):
        _drive(_driver(arm, manage_desk=False)._arm(StopFlag(), clock).move_to(HOME, None), clock)

    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1


def test_a_sync_move_answers_once_the_arm_is_there(world):
    """What a sync move adds over a command: something to wait on that means the arm is in place."""
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


def test_a_move_the_world_stops_under_is_handed_back_to_its_asker(world):
    """A stop ends the travel with no arrival to report, and silence would hold the asker for good."""
    arm = FakeArm(HOME)
    driver = _driver(arm, manage_desk=False)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening reset
        next(loop)
    arm.polls_to_reach = 1000  # the move is still travelling when the stop lands
    answer = _mover(world, driver)(command.JointPosition(JOGGED))
    next(loop)
    assert not answer.done()

    stop.stopped = True
    for _ in range(5):
        if answer.done():
            break
        next(loop)

    with pytest.raises(MoveAbandoned):
        answer.result()


def test_a_sync_move_the_arm_cannot_make_fails_the_asker(world):
    """A move that stops advancing is the asker's failure to hear about."""
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
    """A stall is not a fault the vendor reports, so without this the arm reads AVAILABLE at a pose nobody
    asked for."""
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
    """The sample before the arriving poll was taken mid-travel, and would read AVAILABLE at the pose the
    arm set out from."""
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
    """The deadline stops the poll loop before it asks again, so a goal that landed just then is unseen."""
    arm = FakeArm(HOME, polls_to_reach=10**9)  # it never lands on a poll of its own
    driver = _driver(arm, manage_desk=False)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    travel = driver._arm(StopFlag(), clock).move_to(JOGGED, None)

    next(travel)  # the first poll: the goal is in flight
    clock.advance(60.0)  # the deadline expires
    arm.goal_status = franka.pf.GoalStatus.REACHED  # and the goal lands in the same moment

    with pytest.raises(StopIteration) as done:
        next(travel)
    assert done.value.value is franka.MoveStatus.ARRIVED
    # Only the goal itself was commanded: an arrival is not answered with a hold at where the arm stands
    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1


def test_a_fault_that_lands_with_the_arrival_reads_error_rather_than_available():
    """The state answering a move reports the arm as the vendor describes it, not as the goal reported."""
    arm = FakeArm(HOME, polls_to_reach=1)  # the first poll of the goal already reports it reached
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    travel = driver._arm(StopFlag(), MockClock()).move_to(JOGGED, None)

    arm.error = 1
    with pytest.raises(StopIteration) as done:
        next(travel)

    assert done.value.value is franka.MoveStatus.ARRIVED
    assert states.emitted[-1][1].status == RobotStatus.ERROR


def test_a_sync_move_that_never_arrives_times_out_and_holds_where_the_arm_stopped(world):
    """A goal the controller never converges on is not an error the vendor reports, so the deadline is what
    ends the move."""
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


def test_a_commands_mode_reaches_the_arm_with_the_gains_it_named(desk):
    """Skipping a mode already running is the vendor's, so the driver hands over every command's."""
    arm = FakeArm(HOME)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)  # init + opening reset
    feed.push(command.JointPosition(positions=JOGGED, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)
    stop.stopped = True
    _drive(loop, clock)

    assert isinstance(arm.modes[0], franka.pf.InternalImpedance), 'the arm did not start in its native law'
    applied = [m for m in arm.modes if isinstance(m, franka.pf.SoftwareImpedance)]
    assert applied, 'the command named a mode the arm was never put under'
    assert applied[-1].kq == list(IMPEDANCE.kq), 'the gains the command named did not reach the arm'


def test_homing_returns_the_arm_to_its_native_law(desk):
    arm = FakeArm(HOME)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)
    feed.push(command.JointPosition(positions=JOGGED, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)
    mark = len(arm.modes)
    feed.push(command.Reset())
    for _ in range(6):
        next(loop)
    stop.stopped = True
    _drive(loop, clock)

    assert isinstance(arm.modes[mark], franka.pf.InternalImpedance)
