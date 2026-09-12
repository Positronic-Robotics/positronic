"""What a leader arm publishes while the operator moves it, and what it asks of its controller."""

from typing import Any

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic import geom
from positronic.drivers.roboarm import command, trossen_leader
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

GRIP_TRAVEL_M = 0.04  # the gripper joint's range, which the arm reports and grip is normalized against
GRIPPER_FRICTION = 5.77  # the friction constant term the station's leaders carry, in N
HELD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, GRIP_TRAVEL_M]


class FakeLeader:
    """The slice of ``TrossenArmDriver`` the leader driver uses, under the test's control.

    ``raises`` is what every call to the arm raises, as a controller that has faulted or a link that has
    dropped does.
    """

    class _Limit:
        def __init__(self, lower: float, upper: float):
            self.position_min, self.position_max = lower, upper

    class _Characteristic:
        def __init__(self, friction_constant_term: float):
            self.friction_constant_term = friction_constant_term

    def __init__(self, positions: list[float] | None = None):
        self.positions = list(positions if positions is not None else HELD)
        self.modes: list[Any] = []
        self.efforts: list[list[float]] = []
        self.moves: list[tuple[list[float], float, bool]] = []
        self.characteristics = [self._Characteristic(0.0)] * 6 + [self._Characteristic(GRIPPER_FRICTION)]
        self.friction_terms: list[float] = []
        self.raises: Exception | None = None
        self.cleaned_up = False

    def get_joint_characteristics(self):
        return self.characteristics

    def set_joint_characteristics(self, characteristics) -> None:
        if self.raises is not None:
            raise self.raises
        self.characteristics = list(characteristics)
        self.friction_terms.append(characteristics[6].friction_constant_term)

    def get_joint_limits(self):
        return [self._Limit(-3.14, 3.14)] * 6 + [self._Limit(0.0, GRIP_TRAVEL_M)]

    def set_all_modes(self, mode) -> None:
        if self.raises is not None:
            raise self.raises
        self.modes.append(mode)

    def set_all_external_efforts(self, efforts, goal_time=2.0, blocking=True) -> None:
        if self.raises is not None:
            raise self.raises
        self.efforts.append(list(efforts))

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        if self.raises is not None:
            raise self.raises
        self.moves.append((list(goal_positions), float(goal_time), bool(blocking)))
        self.positions = list(goal_positions)

    def get_all_positions(self):
        if self.raises is not None:
            raise self.raises
        return list(self.positions)

    def cleanup(self) -> None:
        self.cleaned_up = True


def build(arm: FakeLeader, **kwargs):
    """The leader over ``arm``, with recorders on what it publishes and the clock under the test."""
    leader = trossen_leader.Leader('192.168.1.3', connect=lambda _ip: arm, **kwargs)
    joints, grips = RecordingEmitter(), RecordingEmitter()
    leader.joints._bind(joints)
    leader.grip._bind(grips)
    return leader, joints, grips, MockClock(), StopFlag()


def drive(leader, clock: MockClock, stop: StopFlag, *, ticks: int) -> None:
    """Run ``ticks`` of the leader's loop, then let it finish the way a stopped run does."""
    run = leader.run(stop, clock)
    for _ in range(ticks):
        next(run)
        clock.advance(0.01)
    stop.stopped = True
    for _ in run:
        pass


def test_the_leader_publishes_the_joints_the_operator_moves_it_to():
    """The arm joints are the whole of what a follower is asked for, so they go out as they read — the
    gripper is not among them, since it is a grip and travels on its own port."""
    arm = FakeLeader()
    leader, joints, _grips, clock, stop = build(arm)

    drive(leader, clock, stop, ticks=3)

    assert len(joints.emitted) == 3
    for _ts, published in joints.emitted:
        np.testing.assert_allclose(published, HELD[:6])


def test_a_leader_nobody_drives_is_left_for_the_hand_that_holds_it():
    """The operator moves this arm. A leader held in position mode fights the hand on it, so the driver
    reads the arm and asks nothing of it."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm)

    drive(leader, clock, stop, ticks=3)

    names = [mode.name for mode in arm.modes]
    assert names[0] == 'external_effort', 'the arm was not freed for the operator to move'
    assert 'position' not in names, f'the leader was driven, not followed: {names}'
    assert not arm.moves, 'the leader was sent somewhere nobody asked for'
    assert names[-1] == 'idle', 'the run left the arm holding itself up'
    assert arm.cleaned_up


def test_a_move_the_session_asks_for_drives_the_leader_and_gives_it_back(world):
    """Both arms travel to the pose the session opens on, so the follower has no gap to take up when it
    copies its leader. The arm is free in the operator's hand again the moment it arrives."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](leader)
    wire_call(world, caller, leader.sync_move)
    target = np.array([0.0, 1.571, 1.178, 0.0, 0.0, 0.0])

    answer = caller(command.JointPosition(target))
    drive(leader, clock, stop, ticks=2)

    answer.result()
    goal, seconds, blocking = arm.moves[-1]
    np.testing.assert_allclose(goal[:6], target)
    assert goal[6] == pytest.approx(HELD[6]), 'the gripper was moved out of the hand that holds it'
    assert blocking, 'the arm was left travelling with the driver reading it as held'
    assert seconds == pytest.approx(np.max(np.abs(target - np.array(HELD[:6]))) / trossen_leader._MOVE_SPEED)
    names = [mode.name for mode in arm.modes]
    assert names[names.index('position') + 1] == 'external_effort', 'the arm was left holding itself'


def test_a_leader_is_not_driven_to_a_pose(world):
    """A leader has no kinematics: the joints it reads are the whole of what its follower is asked for.
    A pose would have to be solved for, and the arm goes on being read either way."""
    arm = FakeLeader()
    leader, joints, _grips, clock, stop = build(arm)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](leader)
    wire_call(world, caller, leader.sync_move)

    answer = caller(command.CartesianPosition(geom.Transform3D()))
    drive(leader, clock, stop, ticks=2)

    with pytest.raises(NotImplementedError):
        answer.result()
    assert joints.emitted, 'a target the leader cannot take ended the run'
    assert not arm.moves


@pytest.mark.parametrize(
    ('position', 'grip'),
    [(0.0, 1.0), (GRIP_TRAVEL_M, 0.0), (GRIP_TRAVEL_M / 2, 0.5), (-0.001, 1.0), (GRIP_TRAVEL_M + 0.001, 0.0)],
    ids=['closed', 'open', 'halfway', 'past closed', 'past open'],
)
def test_the_trigger_reads_as_the_grip_the_follower_speaks(position: float, grip: float):
    """positronic speaks a grip where 1 is closed, and the trigger reads a little past its range at either
    end — which saturates rather than asking the follower for a grip it has no room for."""
    arm = FakeLeader([*HELD[:6], position])
    leader, _joints, grips, clock, stop = build(arm)

    drive(leader, clock, stop, ticks=1)

    assert grips.emitted[0][1] == pytest.approx(grip)


def test_a_leader_pushes_back_with_nothing_until_the_follower_says_what_it_holds():
    """Zero external effort is gravity compensation: the controller holds the arm's own weight and the
    operator moves it freely. That is what a rig without force feedback runs."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm)

    drive(leader, clock, stop, ticks=2)

    assert arm.efforts, 'the arm was never told what to push with'
    for asked in arm.efforts:
        np.testing.assert_allclose(asked, np.zeros(7))


def test_force_feedback_pushes_back_what_the_follower_is_holding():
    """What the operator feels is the follower's own effort, reversed: the follower pushes into the world
    and the leader pushes into the hand."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm, force_feedback_gain=0.1)
    held = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.5])
    efforts = ManualCommandReceiver()
    efforts.push(held)
    leader.follower_efforts._bind(efforts)

    drive(leader, clock, stop, ticks=2)

    np.testing.assert_allclose(arm.efforts[0], -0.1 * held)


def test_the_gripper_friction_the_operator_asks_for_stands_only_for_the_run():
    """The term is the arm's configuration, and it outlives the process that set it. A run that leaves it
    raised hands the next one an arm that is not the arm it was calibrated as."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm, gripper_friction_constant=8.0)

    drive(leader, clock, stop, ticks=2)

    assert arm.friction_terms == [8.0, GRIPPER_FRICTION], 'the run did not hand the gripper back as it took it'


def test_an_arm_asked_for_no_gripper_friction_keeps_what_it_was_calibrated_with():
    """Every arm ships with its own calibration, and a station that names no preference has none to state."""
    arm = FakeLeader()
    leader, _joints, _grips, clock, stop = build(arm)

    drive(leader, clock, stop, ticks=2)

    assert not arm.friction_terms, 'the run wrote a characteristic nobody asked for'
    assert arm.characteristics[6].friction_constant_term == GRIPPER_FRICTION


def test_a_leader_that_stops_being_read_does_not_end_the_session():
    """One arm of a rig going quiet is not a reason to lose the episode the rest of it is recording. The
    leader complains, holds off, and takes the arm back when it answers again."""
    arm = FakeLeader()
    leader, joints, _grips, clock, stop = build(arm)
    run = leader.run(stop, clock)

    next(run)
    arm.raises = RuntimeError('link down')
    for _ in range(5):
        next(run)
        clock.advance(0.01)
    published_while_down = len(joints.emitted)

    arm.raises = None
    clock.advance(trossen_leader._RECOVER_EVERY_S)
    for _ in range(3):
        next(run)
        clock.advance(0.01)

    assert published_while_down == 1, 'the leader went on publishing an arm it could not read'
    assert len(joints.emitted) > published_while_down, 'the leader never took the arm back'
    stop.stopped = True
    for _ in run:
        pass
