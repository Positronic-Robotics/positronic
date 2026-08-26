import logging
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
from positronic.drivers.utils import MoveAbandoned
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

PARK = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
JOGGED = PARK + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
IMPEDANCE = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
# What the control box answers for a safe input, in its own words.
CLEAR = 'Not triggered (Motion permitted)'
STOPPED = 'Triggered (Motion prohibited)'


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

    ``goal_status`` pins the reported status, so a move that never lands can be scripted; ``raises``, once
    set, is what every call but ``stop`` raises, and ``ik_raises`` what only the solver raises; ``error`` is
    the vendor fault flag every state carries.
    """

    def __init__(self, q, *, polls_to_reach: int = 2, goal_status: 'franka.pf.GoalStatus | None' = None):
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
    """In-memory ``Desk``: records that the session opened the brakes and released control, and reports
    whatever ``safe_inputs`` holds."""

    def __init__(self):
        self.prepared = False
        self.released = False
        self.safe_inputs = dict.fromkeys(('x31', 'x32', 'x33', 'x4'), CLEAR)

    def __enter__(self) -> 'FakeDesk':
        return self

    def __exit__(self, *exc_info) -> bool:
        self.released = True
        return False

    def prepare(self) -> None:
        self.prepared = True

    def _authenticate(self) -> None:
        pass

    def safety_status(self) -> dict[str, Any]:
        return {franka.SAFE_INPUT_STATE: dict(self.safe_inputs)}


@pytest.fixture
def desk(monkeypatch) -> FakeDesk:
    monkeypatch.setenv(franka.DESK_USER_ENV, 'user')
    monkeypatch.setenv(franka.DESK_PASSWORD_ENV, 'password')
    session = FakeDesk()
    monkeypatch.setattr(franka, 'Desk', lambda *credentials: session)
    return session


def _driver(arm: FakeArm, **kwargs) -> franka.Robot:
    robot = franka.Robot('192.0.2.1', **kwargs)
    robot._robot = arm  # `_vendor` hands back an already-set handle, which is how the fake arm gets in
    return robot


def _drive(loop, clock: MockClock | None = None) -> None:
    """Pump a driver loop to exhaustion, standing in for the world by advancing ``clock`` through each wait."""
    clock = clock or MockClock()
    for wait in loop:
        if isinstance(wait, pimm.Sleep):
            clock.advance(wait.seconds)


def _arm(driver: franka.Robot, clock: MockClock) -> franka._Arm:
    """The driver's arm, watching the safe inputs its own configuration reaches."""
    return driver._arm(StopFlag(), clock, driver._safe_inputs())


def _run_move(travel, clock: MockClock) -> franka.MoveStatus:
    """Pump a move to its end, advancing ``clock`` through each wait, and hand back how it ended."""
    try:
        while True:
            wait = next(travel)
            if isinstance(wait, pimm.Sleep):
                clock.advance(wait.seconds)
    except StopIteration as done:
        return done.value


def _drive_park(driver: franka.Robot, arm: FakeArm) -> MockClock:
    """Park ``arm`` under a clock that moves only by the waits the park itself asks for."""
    clock = MockClock()
    _drive(_arm(driver, clock).park(), clock)
    return clock


def _mover(world: pimm.World, driver: franka.Robot) -> pimm.calls.Caller[command.CommandType, None]:
    """A caller on ``driver.sync_move``, for a test that pumps its generator rather than running a World."""
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


def test_park_drives_the_arm_to_the_park_pose():
    arm = FakeArm(JOGGED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    np.testing.assert_allclose(arm.targets, [PARK])
    np.testing.assert_allclose(arm.q, PARK)


def test_the_park_waits_by_yielding_rather_than_blocking():
    """A driver's waits are the world's to honour, teardown included: the park asks for them, never sleeps."""
    arm = FakeArm(JOGGED, polls_to_reach=3)

    commands = list(_arm(_driver(arm, manage_desk=False), MockClock()).park())

    assert commands and all(isinstance(command, pimm.Sleep | pimm.Yield) for command in commands)


def test_park_gives_up_when_the_goal_stops_advancing():
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    assert arm.calls.count(Call.GOAL) == 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_gives_up_when_the_arm_does_not_arrive_in_time():
    arm = FakeArm(JOGGED, polls_to_reach=10**9)
    clock = MockClock()
    parking = _arm(_driver(arm, manage_desk=False), clock)
    budget = parking._travel_s(JOGGED, PARK)

    _drive(parking.park(), clock)

    # It waits out the travel the pose is worth and gives up within a poll of it.
    assert budget <= clock.now() < budget + 0.01
    assert arm.calls.count(Call.GOAL) > 1
    np.testing.assert_allclose(arm.q, JOGGED)


def test_park_swallows_a_robot_that_fails_mid_move():
    arm = FakeArm(JOGGED)
    arm.raises = RuntimeError('libfranka: connection lost')

    _drive_park(_driver(arm, manage_desk=False), arm)

    np.testing.assert_allclose(arm.q, JOGGED)


def test_the_driver_puts_the_arm_at_the_park_pose_when_it_takes_control():
    """Both ends of a run leave the arm at the same pose, so a run starts from where the last one left off."""
    arm = FakeArm(JOGGED)
    loop = _driver(arm, manage_desk=False).run(StopFlag(), MockClock())

    for _ in range(2):  # through the opening move
        next(loop)

    np.testing.assert_allclose(arm.targets, [PARK])
    np.testing.assert_allclose(arm.q, PARK)


def test_a_command_the_arm_cannot_reach_leaves_the_running_law_alone(desk):
    """A rejected command must not half-apply: the arm would hold its old target under new dynamics."""
    arm = FakeArm(PARK)
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


def test_the_law_changes_only_where_the_target_is_published(desk, world):
    """A switch with anything between it and the target can leave the arm holding its last one under it."""
    arm = FakeArm(PARK)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    move = _mover(world, driver)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)  # init + the opening move
    mark = len(arm.calls)
    feed.push(command.JointPosition(positions=JOGGED, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)
    answer = move(command.JointPosition(positions=PARK))  # a travel switches law too, from inside the driver
    for _ in range(20):
        if answer.done():
            break
        next(loop)

    switches = [i for i, c in enumerate(arm.calls[mark:], start=mark) if c is Call.SET_CONTROL_MODE]
    assert switches, 'the commands applied no mode at all'
    assert all(arm.calls[i + 1] is Call.SET_TARGET_JOINTS for i in switches), arm.calls[mark:]


def test_a_joint_target_the_vendor_would_refuse_leaves_the_running_law_alone(desk):
    """A joint command is passed straight through, so what the vendor rejects has to be caught here."""
    arm = FakeArm(PARK)
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
    """The park pose is far off, and only the native law shapes the reference on the way there."""
    arm = FakeArm(JOGGED)

    _drive_park(_driver(arm, manage_desk=False), arm)

    assert isinstance(arm.modes[0], franka.pf.InternalImpedance)


def test_teardown_parks_the_arm_before_stopping_control(desk):
    arm = FakeArm(PARK)
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
    np.testing.assert_allclose(arm.targets[-1], PARK)
    np.testing.assert_allclose(arm.q, PARK)
    assert desk.prepared and desk.released


def test_teardown_stops_control_and_releases_desk_when_parking_fails(desk):
    arm = FakeArm(PARK)
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
    arm = FakeArm(PARK)
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


def test_a_stop_during_the_opening_move_ends_the_run_without_a_fault(desk):
    """The event that ends the world also cancels the in-flight goal, so a poll taken after the stop
    reports failure. Reading it would turn a clean shutdown into a control fault — which skips the park."""
    arm = FakeArm(JOGGED, polls_to_reach=10**9)  # the opening move never lands on its own
    stop = StopFlag()
    clock = MockClock()
    loop = _driver(arm).run(stop, clock)

    next(loop)  # suspended inside the opening move's travel
    stop.stopped = True
    arm.goal_status = franka.pf.GoalStatus.ABORTED

    _drive(loop, clock)

    assert arm.calls[-1] == Call.STOP


def test_an_arm_that_will_not_park_reads_error_rather_than_ending_the_run():
    """The driver's own move is the one that can fail before a caller exists to hear about it, so the run
    goes on and the arm reads as it is."""
    arm = FakeArm(JOGGED, goal_status=franka.pf.GoalStatus.ABORTED)  # the opening move never lands
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
    """What a sync move adds over a command: something to wait on that means the arm is in place."""
    arm = FakeArm(PARK, polls_to_reach=3)  # more than one poll, so an answer cannot land in the asking round
    driver = _driver(arm, manage_desk=False)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening move
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
    arm = FakeArm(PARK)
    driver = _driver(arm, manage_desk=False)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening move
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
    arm = FakeArm(PARK)
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
    arm = FakeArm(PARK)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    move = _mover(world, driver)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening move
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
    answer = move(command.JointPosition(PARK))
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
    arm = FakeArm(PARK, polls_to_reach=3)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening move
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
    arm = FakeArm(PARK, polls_to_reach=10**9)  # it never lands on a poll of its own
    driver = _driver(arm, manage_desk=False)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    travel = _arm(driver, clock).move_to(JOGGED, None)

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
    arm = FakeArm(PARK, polls_to_reach=1)  # the first poll of the goal already reports it reached
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    travel = _arm(driver, MockClock()).move_to(JOGGED, None)

    arm.error = 1
    with pytest.raises(StopIteration) as done:
        next(travel)

    assert done.value.value is franka.MoveStatus.ARRIVED
    assert states.emitted[-1][1].status == RobotStatus.ERROR


def test_a_sync_move_that_never_arrives_times_out_and_holds_where_the_arm_stopped(world):
    """A goal the controller never converges on is not an error the vendor reports, so the deadline is what
    ends the move."""
    arm = FakeArm(PARK)
    driver = _driver(arm, manage_desk=False)
    states = RecordingEmitter()
    driver.state._bind(states)
    move = _mover(world, driver)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(2):  # through the opening move, which still lands
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
    np.testing.assert_allclose(arm.targets[-1], PARK)
    np.testing.assert_allclose(arm.targets[-2], JOGGED)
    # Published before the asker heard: a caller that starts recovering must not read the arm as available
    assert states.emitted[-1][1].status == RobotStatus.ERROR


def test_a_reading_the_driver_does_not_recognise_counts_as_a_triggered_safe_input():
    """The gate opens only on a reading that says motion is permitted, so an unknown one stops nothing."""
    assert not franka._triggered(CLEAR)
    assert franka._triggered(STOPPED)
    assert franka._triggered('a phrase this control box has never sent')


def test_a_move_a_safe_input_stopped_is_made_again_once_it_clears(desk):
    """A stop that clears on its own is the intermittent fault, and the arm goes back to the same target."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    desk.safe_inputs['x31'] = STOPPED
    travel = _arm(driver, clock).move_to(JOGGED, None)

    assert isinstance(next(travel), pimm.Sleep), 'the move failed rather than waiting for the safe input'
    desk.safe_inputs['x31'] = CLEAR
    arm.goal_status = None

    assert _run_move(travel, clock) is franka.MoveStatus.ARRIVED
    np.testing.assert_allclose(arm.q, JOGGED)
    np.testing.assert_allclose(arm.targets, [JOGGED, JOGGED], err_msg='the second move went somewhere else')
    assert arm.calls.count(Call.RECOVER_FROM_ERRORS) == 1


def test_a_safe_input_that_stays_triggered_ends_the_move(desk):
    """A real stop latches the input, so one that never clears is a person's and the move fails as it did."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    desk.safe_inputs['x31'] = STOPPED
    travel = _arm(driver, clock).move_to(JOGGED, None)

    with pytest.raises(RuntimeError, match='stopped short'):
        _run_move(travel, clock)

    assert clock.now() >= franka._Arm._SAFE_STOP_WAIT_S, 'the move gave up before it had waited out the stop'
    assert arm.calls.count(Call.RECOVER_FROM_ERRORS) == 0, 'a latched stop was answered with a recovery'
    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1, 'a latched stop was answered with a fresh target'


def test_a_world_that_comes_down_while_a_safe_input_is_triggered_ends_the_move(desk):
    """A shutdown must not wait out the whole interval a triggered safe input is given."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    stop = StopFlag()
    clock = MockClock()
    desk.safe_inputs['x31'] = STOPPED
    travel = driver._arm(stop, clock, driver._safe_inputs()).move_to(JOGGED, None)

    assert isinstance(next(travel), pimm.Sleep)
    stop.stopped = True

    with pytest.raises(RuntimeError, match='stopped short'):
        _run_move(travel, clock)

    assert clock.now() < franka._Arm._SAFE_STOP_WAIT_S, 'the shutdown waited out the safe input'


def test_a_move_the_arm_refuses_with_every_safe_input_clear_fails_at_once(desk):
    """Only a safe input earns the wait: every other fault reaches the caller the moment it happens."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    travel = _arm(driver, clock).move_to(JOGGED, None)

    with pytest.raises(RuntimeError, match='stopped short'):
        _run_move(travel, clock)

    assert clock.now() == 0.0, 'the move waited on a stop that never happened'
    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1


def test_a_move_is_not_made_again_where_nothing_reads_the_safe_inputs():
    """Without Desk the driver has no reading to attribute a failure to, and keeps the outcome it had."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm, manage_desk=False)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    travel = _arm(driver, clock).move_to(JOGGED, None)

    with pytest.raises(RuntimeError, match='stopped short'):
        _run_move(travel, clock)

    assert clock.now() == 0.0
    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1


def test_a_safe_input_that_stops_every_attempt_gives_the_move_up(desk):
    """The recovery is capped, so a control box tripping over and over ends the move rather than looping."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    readings = iter([{'x31': STOPPED}, {'x31': CLEAR}] * (franka._Arm._SAFE_STOP_RETRIES + 2))
    desk.safety_status = lambda: {franka.SAFE_INPUT_STATE: next(readings)}
    travel = _arm(driver, clock).move_to(JOGGED, None)

    with pytest.raises(RuntimeError, match='stopped short'):
        _run_move(travel, clock)

    assert arm.calls.count(Call.SET_TARGET_JOINTS) == 1 + franka._Arm._SAFE_STOP_RETRIES


def test_the_driver_logs_a_safe_input_that_changes(desk, caplog):
    """One stamped line in the run's own log says when the control box prohibited motion, and when it
    stopped: without it the reason sits only in the control box's separate log, on a separate clock."""
    caplog.set_level(logging.INFO)
    watch = _driver(FakeArm(PARK))._safe_inputs()

    watch.sample()
    desk.safe_inputs['x31'] = STOPPED
    watch.sample()
    desk.safe_inputs['x31'] = CLEAR
    watch.sample()

    assert watch.trips == 1
    assert "safe inputs ['x31'] are triggered" in caplog.text
    assert 'permits motion' in caplog.text


def test_the_driver_logs_the_move_the_arm_refused_with_the_reason_it_gave(desk, caplog):
    """libfranka prints its own rejection from the control thread, unstamped and outside Python."""
    arm = FakeArm(PARK, goal_status=franka.pf.GoalStatus.ABORTED)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    clock = MockClock()
    watching = _arm(driver, clock)

    watching.note_refusals()
    watching.note_refusals()

    assert caplog.text.count('The arm refused a move: scripted') == 1, 'the wall of refusals was logged in full'


def test_a_run_carries_on_after_a_safe_input_stopped_a_move(desk, world):
    """What the recovery is for: the failure that ends the run reaches the driver through a sync move."""
    arm = FakeArm(PARK)
    driver = _driver(arm)
    driver.state._bind(RecordingEmitter())
    move = _mover(world, driver)
    clock = MockClock()
    loop = driver.run(StopFlag(), clock)

    for _ in range(3):  # init + the opening move
        next(loop)
    desk.safe_inputs['x31'] = STOPPED
    arm.goal_status = franka.pf.GoalStatus.ABORTED
    answer = move(command.JointPosition(JOGGED))
    next(loop)  # into the move, which the safe input stops
    desk.safe_inputs['x31'] = CLEAR
    arm.goal_status = None
    for _ in range(20):
        if answer.done():
            break
        next(loop)

    answer.result()
    np.testing.assert_allclose(arm.q, JOGGED)
    next(loop)  # and the loop goes on rather than raising


def test_a_commands_mode_reaches_the_arm_with_the_gains_it_named(desk):
    """Skipping a mode already running is the vendor's, so the driver hands over every command's."""
    arm = FakeArm(PARK)
    driver = _driver(arm)
    feed = ManualCommandReceiver()
    driver.commands._bind(feed)
    stop = StopFlag()
    clock = MockClock()
    loop = driver.run(stop, clock)

    for _ in range(3):
        next(loop)  # init + the opening move
    feed.push(command.JointPosition(positions=JOGGED, mode=IMPEDANCE))
    for _ in range(2):
        next(loop)
    stop.stopped = True
    _drive(loop, clock)

    assert isinstance(arm.modes[0], franka.pf.InternalImpedance), 'the arm did not start in its native law'
    applied = [m for m in arm.modes if isinstance(m, franka.pf.SoftwareImpedance)]
    assert applied, 'the command named a mode the arm was never put under'
    assert applied[-1].kq == list(IMPEDANCE.kq), 'the gains the command named did not reach the arm'


def test_a_command_pinning_no_mode_returns_the_arm_to_its_native_law(desk):
    """A mode is pinned per command, so one that names none does not inherit what the last one ran under."""
    arm = FakeArm(PARK)
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
    feed.push(command.JointPosition(positions=PARK))
    for _ in range(2):
        next(loop)
    stop.stopped = True
    _drive(loop, clock)

    assert isinstance(arm.modes[mark], franka.pf.InternalImpedance)
