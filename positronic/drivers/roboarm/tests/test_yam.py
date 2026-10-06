import dataclasses
import logging
import types
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any

import numpy as np
import pytest

import pimm
import positronic.cfg.embodiment as embodiment_cfg
import positronic.cfg.hardware.roboarm as roboarm_cfg
from pimm.tests.testing import MockClock, wire_call
from positronic import keys
from positronic.cfg import video_encoder
from positronic.cfg.eval.real import yam as yam_eval
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.roboarm.yam import driver as yam
from positronic.eval import Embodiment
from positronic.eval import keys as eval_keys
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

PARK = np.zeros(6)
RAISED = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
DEFAULT_TUNING = yam.PARK_SETTLE
OPEN, CLOSED = 0.0, 1.0
_GRIP_TOL = 0.05
STATION_GRAVITY_COMP = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0]


class FakeYam(yam._FakeYam):
    """A ``_FakeYam`` that misses its command the two ways a real chain does.

    ``bias`` offsets every joint. ``gives_back`` is the fraction of the way from ``floats_at`` to the command
    that the chain travels. ``stops`` places the lower stops of joints 2 and 3.
    """

    def __init__(self):
        super().__init__()
        self.bias = np.zeros(6)
        self.gives_back = 1.0
        self.floats_at = np.zeros(6)
        self.stops = np.zeros(2)
        self.stuck = False
        self.targets = []
        self.released_at = []
        self.closed = False

    def command_joint_pos(self, joint_pos):
        self.targets.append(joint_pos.copy())
        if not self.stuck:
            position = joint_pos.copy()
            position[:6] += self.bias
            position[:6] = self.floats_at + self.gives_back * (position[:6] - self.floats_at)
            position[1:3] = np.maximum(position[1:3], self.stops)
            super().command_joint_pos(position)

    def zero_torque_mode(self):
        self.released_at.append(self._pos.copy())
        super().zero_torque_mode()

    def close(self):
        assert self.released_at, 'connection closed before releasing torque'
        self.closed = True


class Site(Enum):
    """Every collaborator call the driver makes while the motors are enabled, as (port, method)."""

    CHAIN_READ = ('vendor', 'get_observations')
    CHAIN_COMMAND = ('vendor', 'command_joint_pos')
    META_EMIT = ('robot_meta', '_emit')
    STATE_EMIT = ('state', '_emit')
    GRIP_EMIT = ('grip', '_emit')
    COMMANDS_READ = ('commands', 'read')
    TARGET_GRIP_READ = ('target_grip', 'read')


# Calls made after torque is released, outside the protected region.
TEARDOWN_CALLS = {('vendor', 'zero_torque_mode'), ('vendor', 'close')}


class Watch:
    """Records every collaborator call, and raises ``error`` at ``site`` once armed."""

    def __init__(self, site=None, persistent=False):
        self.site, self.persistent, self.armed = site, persistent, False
        self.error = OSError('collaborator failed')
        self.seen = set()

    def raise_at(self, call):
        self.seen.add(call)
        if self.armed and self.site is not None and call == self.site.value:
            self.armed = self.persistent
            # A fresh error per raise: raising one object again grows its traceback.
            raise type(self.error)(*self.error.args) if self.persistent else self.error


class Watched:
    """Forwards to ``target``, reporting each method call on ``port`` to ``watch`` first."""

    def __init__(self, target, port, watch):
        self._target, self._port, self._watch = target, port, watch

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self._watch.raise_at((self._port, name))
            return attr(*args, **kwargs)

        return call


class Rig:
    def __init__(self, park_tuning=DEFAULT_TUNING, move_tuning=yam.MOVE_SETTLE, watch=None):
        self.vendor = FakeYam()
        self.commands = ManualCommandReceiver()
        self.grip = ManualCommandReceiver()
        self.states = RecordingEmitter()
        ports: dict[str, Any] = {
            'vendor': self.vendor,
            'commands': self.commands,
            'target_grip': self.grip,
            'state': self.states,
        }
        ports |= {'grip': RecordingEmitter(), 'robot_meta': RecordingEmitter()}
        if watch is not None:
            ports = {port: Watched(target, port, watch) for port, target in ports.items()}
        self.driver = yam.Robot(
            connect=lambda channel, sim, gravity_comp_factor: ports['vendor'],
            park_after_idle_s=1.0,
            park_tuning=park_tuning,
            move_tuning=move_tuning,
        )
        self.clock = MockClock()
        self.driver.commands._bind(ports['commands'])
        self.driver.target_grip._bind(ports['target_grip'])
        self.driver.state._bind(ports['state'], clock=self.clock)
        self.driver.grip._bind(ports['grip'], clock=self.clock)
        self.driver.robot_meta._bind(ports['robot_meta'], clock=self.clock)
        self.stop = StopFlag()
        self.loop = self.driver.run(self.stop, self.clock)

    def tick(self, seconds=0.01):
        deadline = self.clock.now() + seconds
        while self.clock.now() < deadline:
            wait = next(self.loop)
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)

    def finish(self, within_s=120.0):
        """Stop the driver and run it to its end. A blocked shutdown never ends, so it fails the test."""
        self.stop.stopped = True
        deadline = self.clock.now() + within_s
        for wait in self.loop:
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)
            assert self.clock.now() < deadline, 'shutdown blocked: the driver kept the arm powered'

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


@pytest.mark.parametrize('object_width', [0.2, 0.6])
def test_idle_parking_preserves_an_obstructed_gripper(rig, monkeypatch, object_width):
    rig.raise_arm()
    send = rig.vendor.command_joint_pos

    def hold_object(joint_pos):
        send(joint_pos)
        rig.vendor._pos[6] = max(rig.vendor._pos[6], object_width)
        rig.vendor._vel[6] = 0.0

    monkeypatch.setattr(rig.vendor, 'command_joint_pos', hold_object)
    rig.grip.push(1.0)
    rig.tick(0.5)
    assert rig.vendor._pos[6] == pytest.approx(object_width)
    rig.tick(5)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor.targets[-1], np.append(PARK, object_width), atol=0.005)
    assert not rig.vendor.released_at
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0], np.append(PARK, object_width), atol=0.005)


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


# A bench chain: a correction lands about two thirds of the way, and joints 2-4 float 66 mrad above the stops.
GIVES_BACK = 0.65
FLOATS_AT = np.array([0.0, 0.066, 0.066, 0.066, 0.0, 0.0])
MAX_CORRECTION = DEFAULT_TUNING.max_correction_rad


def _asked(rig):
    return np.array([target[:6] for target in rig.vendor.targets])


def test_parking_closes_a_steady_servo_gap_before_releasing(rig):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert np.min(_asked(rig)) >= -MAX_CORRECTION
    assert rig.vendor.closed


def test_a_chain_that_gives_back_part_of_a_correction_still_lands_on_its_stops(rig):
    """Corrections from the latest gap alone oscillate by about 14 mrad on this chain and never land."""
    rig.raise_arm()
    rig.vendor.gives_back = GIVES_BACK
    rig.vendor.floats_at = FLOATS_AT
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=DEFAULT_TUNING.tolerance_rad)
    assert rig.vendor.closed


def test_a_gap_wider_than_the_correction_bound_keeps_the_arm_powered(rig, caplog):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.0, 0.12, 0.0, 0.0, 0.0])
    rig.stop.stopped = True
    rig.tick(30)
    assert 'Parking failed; arm still powered' in caplog.text
    assert not rig.vendor.released_at
    assert np.min(_asked(rig)) >= -MAX_CORRECTION


def test_a_bench_with_a_wider_gap_is_tuned_not_edited():
    rig = Rig(dataclasses.replace(DEFAULT_TUNING, max_correction_rad=0.2))
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.0, 0.12, 0.0, 0.0, 0.0])
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


class NoisyReadings:
    """Readings of a still chain, with a velocity spike on every other read and joint 1 alternating by
    ``wobble_rad``."""

    def __init__(self, rig, monkeypatch, *, spike_rad_s=0.0, wobble_rad=0.0):
        self.read, self.spike_rad_s, self.wobble_rad, self.reads = (
            rig.vendor.get_observations,
            spike_rad_s,
            wobble_rad,
            0,
        )
        monkeypatch.setattr(rig.vendor, 'get_observations', self)

    def __call__(self):
        obs = self.read()
        self.reads += 1
        sign = 1.0 if self.reads % 2 else -1.0
        obs[yam._JOINT_VEL] = obs[yam._JOINT_VEL] + (self.spike_rad_s if sign > 0 else 0.0)
        obs[yam._JOINT_POS][0] += sign * self.wobble_rad / 2
        return obs


def test_velocity_spikes_at_rest_do_not_stop_a_park(monkeypatch):
    """A real chain reports speed spikes of 0.1-0.3 rad/s at rest."""
    rig = Rig()
    rig.raise_arm()
    NoisyReadings(rig, monkeypatch, spike_rad_s=0.3)
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


@pytest.mark.parametrize(
    ('min_ramp_s', 'still_time_s'), [(DEFAULT_TUNING.min_ramp_s, DEFAULT_TUNING.still_time_s), (4.0, 1.0)]
)
def test_the_shortest_ramp_and_the_still_window_are_tuned_per_arm(min_ramp_s, still_time_s):
    tuning = dataclasses.replace(DEFAULT_TUNING, min_ramp_s=min_ramp_s, still_time_s=still_time_s)
    rig = Rig(tuning)
    rig.raise_arm()
    stopped = rig.clock.now()
    rig.finish()
    ramp_s = max(min_ramp_s, float(np.max(RAISED)) / tuning.max_speed_rad_s)
    assert ramp_s + still_time_s <= rig.clock.now() - stopped < ramp_s + still_time_s + 0.5
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


@pytest.mark.parametrize(('still_position_rad', 'parks'), [(DEFAULT_TUNING.still_position_rad, False), (0.005, True)])
def test_an_arm_with_noisy_position_readings_is_tuned_not_edited(monkeypatch, still_position_rad, parks):
    rig = Rig(dataclasses.replace(DEFAULT_TUNING, still_position_rad=still_position_rad))
    rig.raise_arm()
    NoisyReadings(rig, monkeypatch, wobble_rad=0.003)
    if parks:
        rig.finish()
        np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    else:
        rig.stop.stopped = True
        rig.tick(30)
        assert not rig.vendor.released_at


@pytest.mark.parametrize(('grip_tolerance', 'arrives'), [(yam.MOVE_SETTLE.grip_tolerance, False), (0.1, True)])
def test_an_arm_whose_fingers_read_off_is_tuned_not_edited(world, monkeypatch, grip_tolerance, arrives):
    rig = Rig(move_tuning=dataclasses.replace(yam.MOVE_SETTLE, grip_tolerance=grip_tolerance))
    rig.tick(4)
    read = rig.vendor.get_observations

    def fingers_read_off():
        obs = read()
        obs[yam._GRIPPER_POS] = obs[yam._GRIPPER_POS] - 0.08
        return obs

    monkeypatch.setattr(rig.vendor, 'get_observations', fingers_read_off)
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer)
    if arrives:
        answer.result()
    else:
        with pytest.raises(TimeoutError):
            answer.result()
    rig.loop.close()


def test_the_hardware_configs_give_each_arm_its_own_tuning():
    wide = dataclasses.replace(DEFAULT_TUNING, max_correction_rad=0.2)
    slow = dataclasses.replace(yam.MOVE_SETTLE, max_speed_rad_s=0.2)
    arm = roboarm_cfg.yam.override(**{
        'park_tuning.max_correction_rad': 0.2,
        'move_tuning.max_speed_rad_s': 0.2,
    }).instantiate()
    assert (arm._park_tuning, arm._move_tuning) == (wide, slow)
    bimanual = embodiment_cfg.yam_bimanual.override(
        cameras={}, **{'park_tuning.left.max_correction_rad': 0.2, 'move_tuning.right.max_speed_rad_s': 0.2}
    )
    arms = [system for system in bimanual.instantiate().control_systems if isinstance(system, yam.Robot)]
    assert [(arm._park_tuning, arm._move_tuning) for arm in arms] == [(wide, yam.MOVE_SETTLE), (DEFAULT_TUNING, slow)]


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


def test_starting_at_zero_still_measures_the_gap_after_commanding_the_target(rig):
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.tick(12)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert np.min(_asked(rig)) < 0.0


def test_ordinary_move_can_arrive_with_an_error_that_parking_closes(world, rig):
    rig.tick(4)
    rig.vendor.bias = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    rig.tick(3.5)
    assert answer.done()
    answer.result()
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_array_equal(rig.vendor.targets[-1][:6], RAISED)
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


def _sync_caller(world, rig):
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    return caller


def _until_answered(rig, answer, within_s=60.0):
    deadline = rig.clock.now() + within_s
    while not answer.done():
        assert rig.clock.now() < deadline, 'the move was never answered'
        rig.tick()


def test_a_blocking_move_opens_the_gripper_and_keeps_it_open(world, rig):
    rig.raise_arm()
    rig.grip.push(0.8)
    rig.tick(0.5)
    assert rig.vendor._pos[6] == pytest.approx(0.2, abs=0.005)
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer)
    answer.result()
    rig.tick(0.5)
    assert rig.vendor._pos[6] == pytest.approx(1.0, abs=0.005)  # the chain's 1 is open
    assert rig.vendor.targets[-1][6] == pytest.approx(1.0)


def test_a_move_the_servo_holds_short_of_its_tolerance_settles_onto_its_target(world, rig):
    """The servo holds joint 3 short by 28.5 mrad, past the 20 mrad tolerance."""
    rig.tick(4)
    rig.vendor.bias = np.array([0.0, 0.0, -0.0285, 0.0, 0.0, 0.0])
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer)
    answer.result()
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE


@pytest.mark.parametrize('distance', [0.5, 1.047, 2.0])
def test_a_blocking_move_is_paced_by_the_distance_it_travels(world, rig, monkeypatch, distance):
    rig.tick(4)
    target = np.array([0.0, distance, distance, 0.0, 0.0, 0.0])
    commands = []
    send = rig.vendor.command_joint_pos

    def record(joint_pos):
        commands.append((rig.clock.now(), joint_pos[:6].copy()))
        send(joint_pos)

    monkeypatch.setattr(rig.vendor, 'command_joint_pos', record)
    started = rig.clock.now()
    answer = _sync_caller(world, rig)(command.JointPosition(target))
    _until_answered(rig, answer)
    answer.result()
    times = np.array([time for time, _ in commands])
    targets = np.array([target for _, target in commands])
    speed = yam.MOVE_SETTLE.max_speed_rad_s
    assert np.all(np.abs(np.diff(targets, axis=0)) <= speed * np.diff(times)[:, None] + 1e-10)
    assert rig.clock.now() - started >= distance / speed


def test_a_blocking_move_never_corrects_past_a_joint_limit(world, rig):
    rig.tick(4)
    rig.vendor.bias = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    near_the_stop = np.array([0.0, 0.01, 1.0, 0.0, 0.0, 0.0])
    sent = len(rig.vendor.targets)
    answer = _sync_caller(world, rig)(command.JointPosition(near_the_stop))
    _until_answered(rig, answer)
    with pytest.raises(TimeoutError):
        answer.result()
    assert min(target[1] for target in rig.vendor.targets[sent:]) >= 0.0


def test_a_streamed_command_reaches_the_chain_unramped_and_uncorrected(rig):
    rig.tick(4)
    rig.vendor.bias = np.array([0.0, 0.0, -0.0285, 0.0, 0.0, 0.0])
    rig.commands.push(command.JointPosition(RAISED))
    rig.tick()
    np.testing.assert_array_equal(rig.vendor.targets[-1][:6], RAISED)
    sent = len(rig.vendor.targets)
    rig.tick(0.5)  # inside the rig's one-second idle limit, so no park takes over
    assert all(np.array_equal(target[:6], RAISED) for target in rig.vendor.targets[sent:])


@pytest.mark.parametrize(('stop_rad', 'releases'), [(0.006, True), (0.02, False)])
def test_an_arm_resting_on_a_stop_above_zero_releases_only_near_the_stow_pose(rig, caplog, stop_rad, releases):
    rig.raise_arm()
    rig.vendor.stops = np.array([0.0, stop_rad])
    if releases:
        rig.finish()
        np.testing.assert_allclose(rig.vendor.released_at[0][:6], [0.0, 0.0, stop_rad, 0.0, 0.0, 0.0], atol=1e-3)
        assert rig.vendor.closed
    else:
        rig.stop.stopped = True
        rig.tick(60)
        assert 'Parking failed; arm still powered' in caplog.text
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
    rig.tick(float(np.max(RAISED)) / DEFAULT_TUNING.max_speed_rad_s + 0.5)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_a_move_that_cannot_reach_its_target_fails_within_its_bound(world, rig):
    rig.tick(4)
    rig.vendor.stuck = True
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    tuning = yam.MOVE_SETTLE
    pass_s = float(np.max(RAISED)) / tuning.max_speed_rad_s + tuning.settle_timeout_s
    _until_answered(rig, answer, within_s=tuning.attempts * pass_s)
    with pytest.raises(TimeoutError, match='rests'):
        answer.result()
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR


def test_world_exit_parks_a_foreground_yam_before_closing():
    vendor = FakeYam()
    driver = yam.Robot(connect=lambda channel, sim, gravity_comp_factor: vendor)
    with pimm.World(virtual_time=True) as world:
        loop = world.start(driver)
        for _ in range(150):
            next(loop)
        vendor._pos[:6] = RAISED
    np.testing.assert_allclose(vendor.released_at[0][:6], PARK, atol=0.005)
    assert vendor.closed


def test_a_foreground_yam_that_faults_parks_before_the_world_reports_it():
    vendor = FakeYam()
    driver = yam.Robot(connect=lambda channel, sim, gravity_comp_factor: vendor)
    read = vendor.get_observations
    armed = []

    def read_once_failing():
        if armed == [True]:
            armed.append(False)
            raise OSError('CAN read failed')
        return read()

    vendor.get_observations = read_once_failing
    with pytest.raises(OSError, match='CAN read failed'):
        with pimm.World(virtual_time=True) as world:
            loop = world.start(driver)
            for _ in range(1000):  # past the startup park, into the command loop
                next(loop)
            vendor._pos[:6] = RAISED
            armed.append(True)
            for _ in range(3000):
                next(loop)
    assert armed == [True, False]
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
    assert np.all(np.abs(np.diff(targets, axis=0)) <= DEFAULT_TUNING.max_speed_rad_s * np.diff(times)[:, None] + 1e-10)
    assert np.all(targets >= np.minimum(start, PARK) - 1e-10)
    assert np.all(targets <= np.maximum(start, PARK) + 1e-10)
    assert rig.clock.now() - started >= distance / DEFAULT_TUNING.max_speed_rad_s
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


@pytest.mark.parametrize('wobble_rad', [0.003, float('nan')])
def test_parking_does_not_release_at_target_while_a_joint_moves_or_reads_invalid(rig, monkeypatch, caplog, wobble_rad):
    rig.tick(3)
    rig.vendor._pos[:6] = PARK
    rig.vendor.stuck = True
    NoisyReadings(rig, monkeypatch, wobble_rad=wobble_rad)
    rig.stop.stopped = True
    rig.tick(12)
    assert 'Parking failed; arm still powered' in caplog.text
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert not rig.vendor.released_at
    assert not rig.vendor.closed


def test_parking_requires_continuously_still_arrival_before_release(rig, monkeypatch):
    rig.tick(3)
    rig.vendor._pos[:6] = PARK
    rig.vendor.stuck = True
    noise = NoisyReadings(rig, monkeypatch, wobble_rad=0.003)
    rig.stop.stopped = True
    rig.tick(3)  # past the ramp; joint 1 wobbles, so the chain is not still
    assert not rig.vendor.released_at
    noise.wobble_rad = 0.0
    rig.vendor._pos[0] = 0.01
    rig.tick(0.1)
    rig.vendor._pos[:6] = PARK
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


class FakeMotorInterface:
    """Stands for i2rt's single-motor CAN interface."""

    def __init__(self, vendor, refuses, **opened_with):
        self.vendor = vendor
        self.refuses = refuses
        self.opened_with = opened_with
        self.off = []
        self.closed = False

    def motor_off(self, motor_id):
        assert self.vendor.closed, 'a motor was disabled before close() joined the control thread'
        if motor_id in self.refuses:
            raise OSError(f'motor {motor_id} did not answer')
        self.off.append(motor_id)

    def close(self):
        self.closed = True


@pytest.fixture
def motors(rig, monkeypatch):
    """Give the rig's chain seven CAN motors, and collect every interface the driver opens to disable them."""
    rig.vendor.motor_chain = types.SimpleNamespace(
        channel='can_arm',
        motor_list=[(motor_id, 'DM4310') for motor_id in range(1, 8)],
        motor_interface=types.SimpleNamespace(control_mode='mit'),
    )
    opened = []
    refuses = set()

    def open_interface(**kwargs):
        opened.append(FakeMotorInterface(rig.vendor, refuses, **kwargs))
        return opened[-1]

    monkeypatch.setattr(yam, 'DMSingleMotorCanInterface', open_interface)
    return opened, refuses


def test_a_parked_shutdown_disables_the_motors_after_it_closes_the_chain(rig, motors):
    opened, _ = motors
    rig.raise_arm()
    rig.finish()
    assert [interface.off for interface in opened] == [[1, 2, 3, 4, 5, 6, 7]]
    assert opened[0].closed
    assert opened[0].opened_with == {'channel': 'can_arm', 'control_mode': 'mit', 'name': 'power-off'}


def test_a_motor_that_will_not_answer_leaves_the_others_disabled(rig, motors, caplog):
    opened, refuses = motors
    refuses.add(3)
    rig.raise_arm()
    rig.finish()
    assert opened[0].off == [1, 2, 4, 5, 6, 7]
    assert '[3]' in caplog.text
    assert opened[0].closed


def test_a_blocked_shutdown_leaves_the_motors_enabled(rig, motors):
    opened, _ = motors
    rig.raise_arm()
    rig.vendor.stuck = True
    rig.stop.stopped = True
    rig.tick(30)
    assert not opened


def test_the_motors_are_disabled_even_when_closing_the_chain_raises(rig, motors, monkeypatch):
    opened, _ = motors
    close = rig.vendor.close

    def close_then_fail():
        close()
        raise OSError('bus close failed')

    monkeypatch.setattr(rig.vendor, 'close', close_then_fail)
    rig.raise_arm()
    with pytest.raises(OSError, match='bus close failed'):
        rig.finish()
    assert [interface.off for interface in opened] == [[1, 2, 3, 4, 5, 6, 7]]


def test_a_chain_with_no_motors_opens_no_interface(rig, monkeypatch):
    opened = []
    monkeypatch.setattr(yam, 'DMSingleMotorCanInterface', lambda **kwargs: opened.append(kwargs))
    rig.raise_arm()
    rig.finish()
    assert not opened
    assert rig.vendor.closed


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

    def connect(channel, sim, gravity_comp_factor):
        vendor = FakeYam()
        connections.append(vendor)
        return vendor

    with pimm.World(virtual_time=True) as world:
        world.start(yam.Robot(connect=connect))
    assert not connections


@pytest.mark.parametrize('failed_read', [1, 2])
def test_sync_call_is_answered_when_idle_parking_setup_read_fails(world, parking_rig, monkeypatch, failed_read):
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](parking_rig.driver)
    wire_call(world, caller, parking_rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    read = parking_rig.vendor.get_observations
    reads = 0
    error = OSError('CAN read failed')

    def fail_during_setup():
        nonlocal reads
        reads += 1
        if reads == failed_read:
            raise error
        return read()

    monkeypatch.setattr(parking_rig.vendor, 'get_observations', fail_during_setup)
    parking_rig.tick()
    assert answer.done()
    with pytest.raises(OSError, match='CAN read failed') as raised:
        answer.result()
    assert raised.value is error
    assert not parking_rig.vendor.released_at
    with pytest.raises(OSError, match='CAN read failed'):
        parking_rig.finish()
    np.testing.assert_allclose(parking_rig.vendor.released_at[0][:6], PARK, atol=0.005)


def test_sync_call_is_answered_when_interrupting_idle_parking_cannot_hold(world, parking_rig, monkeypatch):
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](parking_rig.driver)
    wire_call(world, caller, parking_rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    error = OSError('CAN write failed')
    raised_once = []

    def fail_hold(joint_pos):
        # A fresh error per raise: raising one object again grows its traceback.
        if raised_once:
            raise OSError(*error.args)
        raised_once.append(True)
        raise error

    monkeypatch.setattr(parking_rig.vendor, 'command_joint_pos', fail_hold)
    parking_rig.tick()
    assert answer.done()
    with pytest.raises(OSError, match='CAN write failed') as raised:
        answer.result()
    assert raised.value is error
    parking_rig.tick(30)
    assert not parking_rig.vendor.released_at
    assert not parking_rig.vendor.closed


def test_every_call_the_driver_makes_while_the_motors_are_on_is_a_site(world):
    """Closes the list below: a new collaborator call fails here until it gets a ``Site`` and its fault test."""
    watch = Watch()
    rig = Rig(watch=watch)
    rig.raise_arm()
    rig.grip.push(0.5)
    rig.tick(0.1)
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer)
    rig.tick(5)  # idle park
    rig.finish()
    assert watch.seen == {site.value for site in Site} | TEARDOWN_CALLS


@pytest.mark.parametrize('site', list(Site))
def test_a_fault_at_any_site_parks_releases_and_is_raised_again(site):
    watch = Watch(site)
    rig = Rig(watch=watch)
    if site is not Site.META_EMIT:  # the metadata goes out once, as the run starts
        rig.raise_arm()
    watch.armed = True
    rig.tick()  # the fault lands in the run, before anything asks the driver to stop
    assert watch.seen >= {site.value} and not watch.armed
    with pytest.raises(OSError, match='collaborator failed') as raised:
        rig.finish(within_s=30)
    assert raised.value is watch.error
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


@pytest.mark.parametrize('site', [Site.CHAIN_READ, Site.CHAIN_COMMAND])
def test_a_fault_that_prevents_a_verified_park_keeps_the_arm_powered(caplog, site):
    watch = Watch(site, persistent=True)
    rig = Rig(watch=watch)
    rig.raise_arm()
    watch.armed = True
    rig.tick(30)
    assert 'Shutdown blocked' in caplog.text
    assert not rig.vendor.released_at
    assert not rig.vendor.closed
    rig.loop.close()


class SettledStateFails(RecordingEmitter):
    """Refuses every state that reports the arm settled; the reports while the arm moves still go out."""

    def _emit(self, data, time: pimm.Time):
        if data.status is not RobotStatus.BUSY:
            raise OSError('state transport closed')
        super()._emit(data, time)


class FailsOnceStopped(RecordingEmitter):
    """Refuses every emit once the rig is asked to stop, standing for a transport that closed at shutdown."""

    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def _emit(self, data, time: pimm.Time):
        if self.stop.stopped:
            raise OSError('transport closed')
        super()._emit(data, time)


def test_a_shutdown_with_every_publish_failing_still_parks_and_releases(caplog):
    rig = Rig()
    rig.raise_arm()
    rig.driver.state._internal[:] = [FailsOnceStopped(rig.stop)]
    rig.driver.grip._internal[:] = [FailsOnceStopped(rig.stop)]
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed
    assert caplog.text.count('Publishing the arm state failed during shutdown') == 1


def test_a_verified_park_releases_even_when_its_report_fails(caplog):
    rig = Rig()
    rig.raise_arm()
    rig.driver.state._internal[:] = [SettledStateFails()]
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed
    assert 'Publishing the arm state failed during shutdown' in caplog.text


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
        embodiment=embodiment_cfg.yam_bimanual.override(cameras={}, video_encoder=video_encoder.libx264_veryfast)
    ).instantiate()
    (trial,) = rig.tasks()

    grips = _a_trial_after_a_closed_trial(rig.embodiment, trial.prepare_args, [f'.{s}' for s in keys.BIMANUAL_ARMS])

    assert all(g is not None and g < _GRIP_TOL for g in grips), grips
    assert len(chains) == 2
    for chain in chains.values():
        assert chain.last_command is not None and chain.last_command[6] == pytest.approx(1.0)  # the chain's 1 is open


def test_a_single_arm_trial_starts_with_the_gripper_open_after_a_trial_that_closed_it(chains):
    rig = embodiment_cfg.yam.override(video_encoder=video_encoder.libx264_veryfast).instantiate()
    start = {eval_keys.ARM: command.JointPosition(np.asarray(roboarm_cfg.YAM_NOMINAL_JOINTS)), eval_keys.GRIPPER: OPEN}

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
    for passed in _gravity_by_channel(monkeypatch, embodiment_cfg.yam_bimanual_yambox).values():
        np.testing.assert_array_equal(passed, STATION_GRAVITY_COMP)


def test_each_bimanual_chain_gets_the_gravity_compensation_of_its_own_arm(monkeypatch):
    left_channel, right_channel = 'can-left', 'can-right'
    left, right = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0], [1.0, 1.2, 1.3, 1.5, 1.0, 1.0]
    rig = embodiment_cfg.yam_bimanual_yambox.override(
        left_channel=left_channel,
        right_channel=right_channel,
        gravity_comp_factor={keys.LEFT_ARM: left, keys.RIGHT_ARM: right},
    )
    factors = _gravity_by_channel(monkeypatch, rig)
    np.testing.assert_array_equal(factors[left_channel], left)
    np.testing.assert_array_equal(factors[right_channel], right)
