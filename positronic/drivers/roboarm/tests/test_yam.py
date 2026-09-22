import logging

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.roboarm import RobotStatus, command, yam
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

PARK = np.zeros(6)
RAISED = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])


class FakeYam(yam._FakeYam):
    def __init__(self):
        super().__init__()
        self.bias = np.zeros(6)
        self.stuck = False
        self.targets = []
        self.released_at = []
        self.closed = False

    def command_joint_pos(self, joint_pos):
        self.targets.append(joint_pos.copy())
        if not self.stuck:
            position = joint_pos.copy()
            position[:6] += self.bias
            position[1:3] = np.maximum(position[1:3], 0.0)
            super().command_joint_pos(position)

    def zero_torque_mode(self):
        self.released_at.append(self._pos.copy())
        super().zero_torque_mode()

    def close(self):
        assert self.released_at, 'connection closed before releasing torque'
        self.closed = True


class Rig:
    def __init__(self):
        self.vendor = FakeYam()
        self.driver = yam.Robot(connect=lambda channel, sim: self.vendor, park_after_idle_s=1.0)
        self.commands = ManualCommandReceiver()
        self.grip = ManualCommandReceiver()
        self.states = RecordingEmitter()
        self.driver.commands._bind(self.commands)
        self.driver.target_grip._bind(self.grip)
        self.driver.state._bind(self.states)
        self.clock = MockClock()
        self.stop = StopFlag()
        self.loop = self.driver.run(self.stop, self.clock)

    def tick(self, seconds=0.01):
        deadline = self.clock.now() + seconds
        while self.clock.now() < deadline:
            wait = next(self.loop)
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)

    def finish(self):
        self.stop.stopped = True
        for wait in self.loop:
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)

    def raise_arm(self):
        while not self.states.emitted or self.states.emitted[-1][1].status == RobotStatus.BUSY:
            self.tick()
        self.commands.push(command.JointPosition(RAISED))
        self.tick(0.5)
        np.testing.assert_allclose(self.vendor._pos[:6], RAISED, atol=0.005)


@pytest.fixture
def rig():
    result = Rig()
    yield result
    result.loop.close()


def test_idle_parks_once_and_a_new_command_moves_the_arm(rig, caplog):
    caplog.set_level(logging.INFO, logger=yam.__name__)
    rig.raise_arm()
    rig.tick(0.5)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    parked = caplog.text.count('Arm parked')
    rig.tick(10)
    assert caplog.text.count('Arm parked') == parked
    assert not rig.vendor.released_at
    rig.raise_arm()


def test_repeated_identical_grip_commands_delay_parking_until_they_stop(rig):
    rig.raise_arm()
    for _ in range(600):
        rig.grip.push(0.0)
        rig.tick()
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(0.8)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_sync_move_starts_idle_time_at_completion(world, rig):
    rig.tick(4)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    while not answer.done():
        rig.tick()
    answer.result()
    rig.tick(0.8)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_arm_and_grip_commands_reset_idle_time(rig):
    rig.raise_arm()
    for _ in range(5):
        rig.commands.push(command.JointPosition(RAISED))
        rig.tick(0.5)
    rig.grip.push(0.7)
    rig.tick(0.8)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert rig.vendor._pos[6] == pytest.approx(0.3, abs=0.005)


def test_measured_motion_does_not_reset_command_idle_time(rig):
    rig.raise_arm()
    rig.vendor._vel[:6] = 0.1
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_disabling_idle_parking_still_parks_on_exit(rig):
    rig.driver._park_after_idle_s = None
    rig.raise_arm()
    rig.tick(60)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


def test_startup_and_shutdown_preserve_the_gripper(rig):
    rig.vendor._pos[:6] = RAISED
    rig.vendor._pos[6] = 0.4
    rig.tick(4)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert rig.vendor._pos[6] == pytest.approx(0.4)
    rig.raise_arm()
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0], np.append(PARK, 0.4), atol=0.005)


def test_parking_does_not_offset_the_target_to_compensate_for_bias(rig, caplog):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.stop.stopped = True
    rig.tick(12)
    assert 'Parking failed; arm still powered' in caplog.text
    assert not rig.vendor.released_at
    assert not rig.vendor.closed
    assert np.min(rig.vendor.targets) >= 0.0
    assert any(np.array_equal(target[:6], PARK) for target in rig.vendor.targets)


def test_failed_shutdown_parking_keeps_the_arm_powered(rig, caplog):
    rig.raise_arm()
    rig.vendor.stuck = True
    rig.stop.stopped = True
    rig.tick(120)
    assert 'Parking failed; arm still powered' in caplog.text
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert not rig.vendor.closed
    assert not rig.vendor.released_at
    np.testing.assert_allclose(rig.vendor.targets[-1][:6], RAISED, atol=0.005)


def test_stop_during_a_move_answers_the_caller_and_parks(world, rig):
    rig.tick(4)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    rig.tick(0.5)
    rig.finish()
    with pytest.raises(yam.MoveAbandoned):
        answer.result()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_idle_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match='park_after_idle_s'):
        yam.Robot(park_after_idle_s=timeout)


def test_starting_at_zero_still_checks_for_bias_after_commanding_the_target(rig):
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.tick(12)
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert np.min(rig.vendor.targets) >= 0.0


def test_ordinary_move_can_arrive_while_parking_rejects_the_same_error(world, rig):
    rig.tick(4)
    rig.vendor.bias = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    rig.tick(2.5)
    assert answer.done()
    answer.result()
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_array_equal(rig.vendor.targets[-1][:6], RAISED)
    rig.stop.stopped = True
    rig.tick(12)
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert not rig.vendor.released_at


def test_parking_accepts_error_within_its_tolerance(rig):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.003, 0.0, 0.0, 0.0, 0.0])
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    np.testing.assert_array_equal(rig.vendor.targets[-1][:6], PARK)
    assert rig.vendor.closed


def test_parking_allows_more_time_than_an_ordinary_move(rig):
    rig.vendor._pos[:6] = RAISED
    rig.vendor.stuck = True
    rig.tick(4)
    assert rig.states.emitted[-1][1].status == RobotStatus.BUSY
    rig.vendor.stuck = False
    rig.tick(3)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_parking_finishes_after_measured_arrival_and_stopping(rig):
    rig.vendor._pos[:6] = RAISED
    rig.tick(2.5)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_ordinary_move_keeps_its_shorter_timeout(world, rig):
    rig.tick(4)
    rig.vendor.stuck = True
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    rig.tick(4)
    assert answer.done()
    with pytest.raises(TimeoutError, match='after 3s'):
        answer.result()


def test_world_exit_parks_a_foreground_yam_before_closing():
    vendor = FakeYam()
    driver = yam.Robot(connect=lambda channel, sim: vendor)
    with pimm.World(virtual_time=True) as world:
        loop = world.start(driver)
        for _ in range(150):
            next(loop)
        vendor._pos[:6] = RAISED
    np.testing.assert_allclose(vendor.released_at[0][:6], PARK, atol=0.005)
    assert vendor.closed


@pytest.mark.parametrize('distance', [0.1, 1.0, 3.0])
def test_parking_command_speed_is_bounded_for_different_distances(rig, monkeypatch, distance):
    rig.tick(3)
    start = distance * np.array([1.0, 1.0, 0.5, -1.0, 0.2, -0.3])
    rig.vendor._pos[:6] = start
    commands = []
    send = rig.vendor.command_joint_pos

    def record(joint_pos):
        commands.append((rig.clock.now(), joint_pos[:6].copy()))
        send(joint_pos)

    monkeypatch.setattr(rig.vendor, 'command_joint_pos', record)
    started = rig.clock.now()
    rig.finish()
    times = np.array([time for time, _ in commands])
    targets = np.array([target for _, target in commands])
    assert np.all(np.abs(np.diff(targets, axis=0)) <= 0.5 * np.diff(times)[:, None] + 1e-10)
    assert np.all(targets >= np.minimum(start, PARK) - 1e-10)
    assert np.all(targets <= np.maximum(start, PARK) + 1e-10)
    assert rig.clock.now() - started >= distance / 0.5
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


@pytest.mark.parametrize('speed', [0.03, -0.03, float('nan'), float('inf')])
def test_parking_does_not_release_at_target_with_moving_or_invalid_velocity(rig, caplog, speed):
    rig.tick(3)
    rig.vendor._pos[:6] = PARK
    rig.vendor.stuck = True
    rig.vendor._vel[0] = speed
    rig.stop.stopped = True
    rig.tick(12)
    assert 'Parking failed; arm still powered' in caplog.text
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert not rig.vendor.released_at
    assert not rig.vendor.closed


@pytest.mark.parametrize(('position_offset', 'velocity'), [(0.01, 0.0), (0.0, 0.03)])
def test_parking_requires_continuously_still_arrival_before_release(rig, position_offset, velocity):
    rig.tick(3)
    rig.vendor._pos[:6] = PARK
    rig.vendor.stuck = True
    rig.vendor._vel[0] = 0.03
    rig.stop.stopped = True
    rig.tick(3)
    assert not rig.vendor.released_at

    rig.vendor._vel[:] = 0.0
    rig.tick(0.1)
    assert not rig.vendor.released_at
    rig.vendor._pos[0] = position_offset
    rig.vendor._vel[0] = velocity
    rig.tick(0.1)
    rig.vendor._pos[:6] = PARK
    rig.vendor._vel[:] = 0.0
    still_since = rig.clock.now()
    rig.tick(0.15)
    assert not rig.vendor.released_at
    rig.finish()
    assert rig.clock.now() - still_since >= 0.2
    assert rig.vendor.closed


def test_shutdown_read_failure_blocks_release_even_after_reading_recovers(rig, monkeypatch, caplog):
    rig.raise_arm()
    rig.stop.stopped = True
    read = rig.vendor.get_observations
    failed = False

    def read_with_one_failure():
        nonlocal failed
        if not failed:
            failed = True
            raise OSError('CAN read failed')
        return read()

    monkeypatch.setattr(rig.vendor, 'get_observations', read_with_one_failure)
    rig.tick(20)
    assert 'Could not verify parking' in caplog.text
    assert not rig.vendor.closed
    assert not rig.vendor.released_at
    np.testing.assert_allclose(rig.vendor.targets[-1][:6], RAISED, atol=0.005)


@pytest.fixture
def blocked_shutdown_rig(rig):
    rig.raise_arm()
    rig.vendor.stuck = True
    rig.stop.stopped = True
    rig.tick(12)
    return rig


def test_failed_read_does_not_end_blocked_shutdown(blocked_shutdown_rig, monkeypatch, caplog):
    def fail_read():
        raise OSError('CAN read failed')

    with monkeypatch.context() as patch:
        patch.setattr(blocked_shutdown_rig.vendor, 'get_observations', fail_read)
        blocked_shutdown_rig.tick(0.05)
    blocked_shutdown_rig.tick(1)
    assert 'Could not hold the arm; shutdown remains blocked' in caplog.text
    assert not blocked_shutdown_rig.vendor.closed
    assert not blocked_shutdown_rig.vendor.released_at


def test_failed_command_does_not_end_blocked_shutdown(blocked_shutdown_rig, monkeypatch, caplog):
    def fail_command(joint_pos):
        raise OSError('CAN write failed')

    with monkeypatch.context() as patch:
        patch.setattr(blocked_shutdown_rig.vendor, 'command_joint_pos', fail_command)
        blocked_shutdown_rig.tick(0.05)
    blocked_shutdown_rig.tick(1)
    assert 'Could not hold the arm; shutdown remains blocked' in caplog.text
    assert not blocked_shutdown_rig.vendor.closed
    assert not blocked_shutdown_rig.vendor.released_at


def test_interrupted_driver_does_not_explicitly_release_torque(rig, caplog):
    rig.raise_arm()
    rig.loop.close()
    assert 'before verified parking' in caplog.text
    assert not rig.vendor.closed
    assert not rig.vendor.released_at


@pytest.fixture
def parking_rig(rig):
    rig.raise_arm()
    rig.tick(0.8)
    assert rig.states.emitted[-1][1].status == RobotStatus.BUSY
    assert 0.1 < rig.vendor._pos[1] < RAISED[1]
    return rig


def test_arm_command_interrupts_idle_parking(parking_rig):
    target = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
    parking_rig.commands.push(command.JointPosition(target))
    parking_rig.tick(0.02)
    np.testing.assert_array_equal(parking_rig.vendor.targets[-1][:6], target)
    assert not parking_rig.vendor.released_at


def test_gripper_command_interrupts_idle_parking_and_holds_arm(parking_rig):
    position = parking_rig.vendor._pos[:6].copy()
    parking_rig.grip.push(0.7)
    parking_rig.tick(0.02)
    np.testing.assert_allclose(parking_rig.vendor.targets[-1][:6], position)
    assert parking_rig.vendor.targets[-1][6] == pytest.approx(0.3)
    assert not parking_rig.vendor.released_at


def test_sync_command_interrupts_idle_parking(world, parking_rig):
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](parking_rig.driver)
    wire_call(world, caller, parking_rig.driver.sync_move)
    target = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
    answer = caller(command.JointPosition(target))
    parking_rig.tick(2.5)
    assert answer.done()
    answer.result()
    np.testing.assert_array_equal(parking_rig.vendor.targets[-1][:6], target)
    assert not parking_rig.vendor.released_at


def test_commands_do_not_interrupt_shutdown_parking(world, rig):
    rig.raise_arm()
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    rig.stop.stopped = True
    rig.tick(0.2)
    rig.commands.push(command.JointPosition(RAISED))
    rig.grip.push(0.7)
    caller(command.JointPosition(RAISED))
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


def test_unstarted_foreground_yam_does_not_connect_on_world_exit():
    connections = []

    def connect(channel, sim):
        vendor = FakeYam()
        connections.append(vendor)
        return vendor

    with pimm.World(virtual_time=True) as world:
        world.start(yam.Robot(connect=connect))
    assert not connections
