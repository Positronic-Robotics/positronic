import dataclasses
import logging
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pimm
import positronic.cfg.embodiment as embodiment_cfg
import positronic.cfg.hardware.roboarm as roboarm_cfg
from pimm.tests.testing import MockClock, wire_call
from positronic import geom, keys, wire
from positronic.cfg.eval.real import yam as yam_eval
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.faults import DeviceFault, FaultKind
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm import yam as yam_package
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.models import GRIPPER_SITE, YAM_JOINT_NAMES, bundled_yam_model
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.roboarm.yam import driver as yam
from positronic.eval import Embodiment, Task
from positronic.eval import keys as eval_keys
from positronic.policy.base import Policy, PolicyRun, Step
from positronic.policy.harness import Harness, Rollout
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

PARK = np.zeros(6)
RAISED = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
DEFAULT_TUNING = yam.PARK_SETTLE
MAX_STREAMED_STEP = yam._Arm._MAX_STREAMED_JOINT_SPEED_RAD_S / yam._CONTROL_HZ
OPEN, CLOSED = 0.0, 1.0
_GRIP_TOL = 0.05
STATION_GRAVITY_COMP = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0]
# The joint ranges of the i2rt YAM arm model, and the limits an i2rt chain reports: those ranges widened by
# 0.15 rad (i2rt get_robot.py:196).
I2RT_YAM_RANGES = np.array([
    [-2.61799, 3.14159],
    [-8.88178e-16, 3.66519],
    [0.0, 3.14159],
    [-1.69297, 1.5708],
    [-1.5708, 1.5708],
    [-2.0944, 2.0944],
])
I2RT_YAM_LIMITS = I2RT_YAM_RANGES + np.array([-0.15, 0.15])
I2RT_GRAVITY_COMP = [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]  # i2rt's own factors for the YAM arm


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
        self.joint_limits = I2RT_YAM_LIMITS.copy()
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
    INFO_READ = ('vendor', 'get_robot_info')
    META_EMIT = ('robot_meta', '_emit')
    ARM_META_EMIT = ('arm_meta', '_emit')
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
    def __init__(self, park_tuning=DEFAULT_TUNING, move_tuning=yam.MOVE_SETTLE, watch=None, park_after_idle_s=1.0):
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
        self.grips = RecordingEmitter()
        self.arm_metas = RecordingEmitter()
        ports |= {'grip': self.grips, 'robot_meta': RecordingEmitter(), 'arm_meta': self.arm_metas}
        if watch is not None:
            ports = {port: Watched(target, port, watch) for port, target in ports.items()}
        self.driver = yam.Robot(
            connect=lambda channel, sim, gravity_comp_factor: ports['vendor'],
            park_after_idle_s=park_after_idle_s,
            park_tuning=park_tuning,
            move_tuning=move_tuning,
        )
        self.clock = MockClock()
        self.driver.commands._bind(ports['commands'])
        self.driver.target_grip._bind(ports['target_grip'])
        self.driver.state._bind(ports['state'], clock=self.clock)
        self.driver.grip._bind(ports['grip'], clock=self.clock)
        self.driver.robot_meta._bind(ports['robot_meta'], clock=self.clock)
        self.driver.arm_meta._bind(ports['arm_meta'], clock=self.clock)
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

    def ready(self):
        """Run until the start move ends."""
        while not self.states.emitted or self.states.emitted[-1][1].status == RobotStatus.BUSY:
            self.tick()

    def rest(self):
        """Run past the start move and the idle park after it, until the arm rests at the parking pose."""
        self.tick(12)
        np.testing.assert_allclose(self.vendor._pos[:6], PARK, atol=0.005)

    def raise_arm(self):
        self.ready()
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
    rig.rest()
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


def test_the_start_raises_the_arm_and_opens_the_gripper(rig):
    rig.vendor._pos[6] = 0.4
    rig.ready()
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)
    assert rig.vendor._pos[6] == pytest.approx(1.0, abs=0.005)  # the chain's 1 is open


def test_an_arm_left_alone_after_the_start_parks_after_the_idle_time(rig):
    rig.ready()
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_an_arm_that_cannot_reach_the_start_pose_reads_error_and_takes_commands(rig, caplog):
    rig.vendor.stuck = True
    rig.ready()
    assert 'did not reach the start pose' in caplog.text
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    rig.vendor.stuck = False
    rig.commands.push(command.JointPosition(RAISED))
    rig.tick(0.5)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)


def test_shutdown_preserves_the_gripper(rig):
    rig.raise_arm()
    rig.grip.push(0.6)
    rig.tick(0.5)
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
    rig.rest()
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
        with pytest.raises(DeviceFault) as answered:
            answer.result()
        assert answered.value.kind is FaultKind.ARM
        assert isinstance(answered.value.__cause__, TimeoutError)
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


def test_the_bimanual_embodiment_states_a_frame_for_each_pose_signal_of_both_arms():
    static_meta = embodiment_cfg.yam_bimanual.override(cameras={}).instantiate().static_meta
    frames = ee_frames({**static_meta, roboarm_keys.URDF: bundled_yam_model()[roboarm_keys.URDF]})
    assert set(frames) == set(static_meta[eval_keys.POSE_SIGNALS])
    for side in (keys.LEFT_ARM, keys.RIGHT_ARM):
        assert frames[f'{keys.ROBOT_STATE}.{side}{keys.EE_POSE_SUFFIX}'].site == GRIPPER_SITE
        assert frames[f'{keys.ROBOT_COMMAND}.{side}{keys.POSE_SUFFIX}'].site == GRIPPER_SITE


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
    rig.rest()
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


def test_parking_from_zero_still_measures_the_gap_after_commanding_the_target(rig):
    rig.rest()
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    sent = len(rig.vendor.targets)
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert np.min(_asked(rig)[sent:]) < 0.0


def test_ordinary_move_can_arrive_with_an_error_that_parking_closes(world, rig):
    rig.rest()
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
    rig.rest()
    rig.vendor.bias = np.array([0.0, 0.0, -0.0285, 0.0, 0.0, 0.0])
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer)
    answer.result()
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE


@pytest.mark.parametrize('distance', [0.5, 1.047, 2.0])
def test_a_blocking_move_is_paced_by_the_distance_it_travels(world, rig, monkeypatch, distance):
    rig.rest()
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
    rig.rest()
    rig.vendor.bias = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    near_the_stop = np.array([0.0, 0.01, 1.0, 0.0, 0.0, 0.0])
    sent = len(rig.vendor.targets)
    answer = _sync_caller(world, rig)(command.JointPosition(near_the_stop))
    _until_answered(rig, answer)
    with pytest.raises(DeviceFault) as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, TimeoutError)
    assert min(target[1] for target in rig.vendor.targets[sent:]) >= I2RT_YAM_RANGES[1, 0] - 1e-12


def test_a_streamed_target_one_tick_of_travel_away_reaches_the_chain_unchanged_and_uncorrected(rig):
    rig.rest()
    rig.vendor.bias = np.array([0.0, 0.0, -0.0285, 0.0, 0.0, 0.0])
    target = np.array([0.0, MAX_STREAMED_STEP, MAX_STREAMED_STEP / 2, 0.0, -MAX_STREAMED_STEP, 0.0])
    rig.commands.push(command.JointPosition(target))
    rig.tick()
    # The cap can trim ~1e-16 rad: the mock clock's elapsed time carries float rounding.
    np.testing.assert_allclose(rig.vendor.targets[-1][:6], target, rtol=0, atol=1e-12)
    sent = len(rig.vendor.targets)
    rig.tick(0.5)  # inside the rig's one-second idle limit, so no park takes over
    assert all(np.array_equal(asked, target) for asked in _asked(rig)[sent:])


def test_a_far_streamed_target_is_approached_in_a_line_at_the_speed_cap(rig):
    rig.rest()
    sent = len(rig.vendor.targets)
    rig.commands.push(command.JointPosition(RAISED))
    rig.tick(0.5)
    asked = _asked(rig)[sent - 1 :]
    steps = np.max(np.abs(np.diff(asked, axis=0)), axis=1)
    assert steps[0] == pytest.approx(MAX_STREAMED_STEP)
    assert np.all(steps <= MAX_STREAMED_STEP * (1 + 1e-9))
    np.testing.assert_array_equal(asked[:, 1], asked[:, 2])
    assert not asked[:, [0, 3, 4, 5]].any()
    np.testing.assert_array_equal(asked[-1], RAISED)


def test_a_late_tick_moves_the_streamed_target_no_faster_than_the_cap(rig, monkeypatch):
    rig.rest()
    sent_at = []
    command_joint_pos = rig.vendor.command_joint_pos

    def timed(joint_pos):
        sent_at.append(rig.clock.now())
        command_joint_pos(joint_pos)

    monkeypatch.setattr(rig.vendor, 'command_joint_pos', timed)
    rig.commands.push(command.JointPosition(RAISED))
    rig.tick(0.05)
    rig.clock.advance(0.0095)  # the loop wakes 9.5 ms late, and its next tick comes 0.5 ms later
    rig.tick(0.05)
    steps = np.max(np.abs(np.diff(_asked(rig)[-len(sent_at) :], axis=0)), axis=1)
    assert np.all(steps <= yam._Arm._MAX_STREAMED_JOINT_SPEED_RAD_S * np.diff(sent_at) + 1e-12)


def test_a_stalled_loop_moves_the_streamed_target_one_tick_at_most_when_it_wakes(rig):
    rig.rest()
    rig.commands.push(command.JointPosition(RAISED))
    rig.tick(0.05)
    sent = len(rig.vendor.targets)
    rig.clock.advance(0.2)  # the loop stalls for twenty ticks
    rig.tick()
    asked = _asked(rig)[sent - 1 :]
    assert np.max(np.abs(np.diff(asked, axis=0))) <= MAX_STREAMED_STEP * (1 + 1e-9)


# A joint step between two commands of a 15 Hz policy, which the cap must pass before the next command
POLICY_STEP_RAD, POLICY_HZ = 0.12, 15.0


def test_the_speed_cap_lands_a_policy_step_before_the_next_command(rig):
    rig.raise_arm()
    target = RAISED + np.array([0.0, 0.0, 0.0, POLICY_STEP_RAD, 0.0, 0.0])
    rig.commands.push(command.JointPosition(target))
    rig.tick(1 / POLICY_HZ)
    np.testing.assert_array_equal(rig.vendor.targets[-1][:6], target)


@pytest.mark.parametrize(
    'past_the_range',
    [
        command.JointPosition(RAISED + np.array([0.0, 0.0, 0.0, 3.0, 0.0, -3.0])),
        command.JointDelta(np.array([0.0, 0.0, 0.0, 3.0, 0.0, -3.0])),
    ],
)
def test_a_streamed_target_past_the_joint_range_stops_at_the_limit(rig, past_the_range):
    rig.raise_arm()
    sent = len(rig.vendor.targets)
    rig.commands.push(past_the_range)
    rig.tick(0.9)
    asked = _asked(rig)[sent:]
    assert np.all((asked >= I2RT_YAM_RANGES[:, 0] - 1e-12) & (asked <= I2RT_YAM_RANGES[:, 1] + 1e-12))
    assert asked[-1][3] == pytest.approx(I2RT_YAM_RANGES[3, 1])
    assert asked[-1][5] == pytest.approx(I2RT_YAM_RANGES[5, 0])


def test_the_joint_range_is_the_range_of_the_i2rt_arm_model():
    rig = Rig(park_after_idle_s=10.0)
    rig.raise_arm()
    for side, past in ((0, I2RT_YAM_RANGES[:, 0] - 0.5), (1, I2RT_YAM_RANGES[:, 1] + 0.5)):
        rig.commands.push(command.JointPosition(past))
        rig.tick(2.0)
        np.testing.assert_allclose(_asked(rig)[-1], I2RT_YAM_RANGES[:, side], atol=1e-12)
    rig.loop.close()


def test_a_cartesian_target_that_needs_joint_4_past_the_mjcf_range_reaches_the_chain(rig):
    rig.raise_arm()
    wrist_down = RAISED.copy()
    wrist_down[3] = -1.62  # past the -1.5708 of the MJCF, inside the -1.69297 of the i2rt arm model
    rig.commands.push(command.CartesianPosition(yam._Kinematics().fk(wrist_down)))
    rig.tick(0.9)
    np.testing.assert_allclose(_asked(rig)[-1], wrist_down, atol=1e-6)


# i2rt halts the chain when a joint reads this far past its limits (i2rt motor_chain_robot.py:254).
I2RT_HALT_PAST_LIMIT_RAD = 0.1
JOINT3 = 2


def test_a_streamed_target_past_the_chain_limit_stops_short_of_where_the_chain_halts(rig):
    rig.raise_arm()
    folded = RAISED.copy()
    folded[JOINT3] = 3.6  # past the chain's range
    sent = len(rig.vendor.targets)
    rig.commands.push(command.JointPosition(folded))
    rig.tick(0.9)
    asked = _asked(rig)[sent:]
    halts_above = I2RT_YAM_LIMITS[JOINT3, 1] + I2RT_HALT_PAST_LIMIT_RAD
    assert np.max(asked[:, JOINT3]) < halts_above - yam._Arm._CHAIN_LIMIT_MARGIN_RAD
    assert asked[-1][JOINT3] == pytest.approx(I2RT_YAM_LIMITS[JOINT3, 1] - yam._Arm._CHAIN_LIMIT_MARGIN_RAD)


def test_a_streamed_target_inside_the_chain_limits_reaches_the_chain_unchanged(rig):
    rig.raise_arm()
    folded = RAISED.copy()
    folded[JOINT3] = 3.0
    rig.commands.push(command.JointPosition(folded))
    rig.tick(0.9)
    np.testing.assert_array_equal(_asked(rig)[-1], folded)


def test_a_blocking_move_past_the_chain_limit_never_sends_a_target_past_it(world, rig):
    rig.raise_arm()
    folded = RAISED.copy()
    folded[JOINT3] = 3.6
    sent = len(rig.vendor.targets)
    answer = _sync_caller(world, rig)(command.JointPosition(folded))
    _until_answered(rig, answer)
    with pytest.raises(DeviceFault):
        answer.result()
    assert np.max(_asked(rig)[sent:, JOINT3]) <= I2RT_YAM_LIMITS[JOINT3, 1] - yam._Arm._CHAIN_LIMIT_MARGIN_RAD + 1e-12


def test_the_park_holds_its_correction_past_the_joint_range(rig):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.0, 0.025, 0.0, 0.0, 0.0])
    rig.tick(10)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    sent = len(rig.vendor.targets)
    rig.tick(0.5)
    held = _asked(rig)[sent:]
    assert np.all(held[:, 2] < yam._Kinematics().lower[2])
    assert all(np.array_equal(asked, held[0]) for asked in held)


def _solver_answers(monkeypatch, posture: np.ndarray, *, success: bool) -> None:
    """Make the LM solver answer ``posture`` from every start, and report ``success``."""

    def solve(model, data, *_args, **_kwargs):
        data.qpos[[model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES]] = posture
        return data.qpos, 0.0, success

    monkeypatch.setattr(yam, 'qpos_from_site_pose', solve)


# Reaching forward and down. At this joint-3 angle, links 2 and 3 are on one line (MJCF joint-4 offset 0.245, 0.06).
STRETCHED = np.array([0.0, 2.9, np.pi - np.arctan2(0.06, 0.245), -0.6, 0.0, 0.0])


def past_full_stretch(kinematics, distance_m: float) -> geom.Transform3D:
    """The pose at ``STRETCHED``, ``distance_m`` farther from the arm base."""
    pose = kinematics.fk(STRETCHED)
    away = pose.translation / np.linalg.norm(pose.translation)
    return geom.Transform3D(pose.translation + distance_m * away, pose.rotation)


def test_ik_takes_an_answer_the_solver_reports_as_failed_when_it_is_inside_the_tolerance(monkeypatch):
    kinematics = yam._Kinematics()
    _solver_answers(monkeypatch, STRETCHED, success=False)
    target = past_full_stretch(kinematics, yam._IK_POS_TOL / 2)
    np.testing.assert_array_equal(kinematics.ik(target, STRETCHED, kinematics.lower, kinematics.upper), STRETCHED)


def test_ik_accepts_a_target_past_full_stretch_inside_the_tolerance_from_every_nearly_equal_start():
    kinematics = yam._Kinematics()
    target = past_full_stretch(kinematics, 0.8 * yam._IK_POS_TOL)
    for k in range(20):
        assert kinematics.ik(target, STRETCHED + k * 1e-12, kinematics.lower, kinematics.upper) is not None


def test_ik_refuses_a_target_farther_past_full_stretch_than_the_tolerance():
    kinematics = yam._Kinematics()
    target = past_full_stretch(kinematics, 5 * yam._IK_POS_TOL)
    assert kinematics.ik(target, STRETCHED, kinematics.lower, kinematics.upper) is None


def test_ik_clamps_a_joint_just_below_its_range_to_its_lower_limit(monkeypatch):
    kinematics = yam._Kinematics()
    asked = RAISED.copy()
    asked[3] = kinematics.lower[3] - 0.002
    _solver_answers(monkeypatch, asked, success=True)
    held = asked.copy()
    held[3] = kinematics.lower[3]
    np.testing.assert_array_equal(kinematics.ik(kinematics.fk(asked), RAISED, kinematics.lower, kinematics.upper), held)


def test_ik_wraps_a_joint_a_full_turn_below_its_range_back_into_it(monkeypatch):
    kinematics = yam._Kinematics()
    inside = RAISED.copy()
    inside[0] = 2.98  # a full turn below is past joint 1's lower limit, and the turn fits its range
    a_turn_below = inside.copy()
    a_turn_below[0] -= 2 * np.pi
    _solver_answers(monkeypatch, a_turn_below, success=True)
    solution = kinematics.ik(kinematics.fk(inside), RAISED, kinematics.lower, kinematics.upper)
    assert solution is not None
    np.testing.assert_allclose(solution, inside)


@pytest.mark.parametrize('positions', [[np.nan, 1.047, 1.047, 0.0, 0.0, 0.0], [0.5]])
def test_a_streamed_target_that_is_not_six_finite_joints_is_refused(rig, caplog, positions):
    rig.raise_arm()
    sent = len(rig.vendor.targets)
    rig.commands.push(command.JointPosition(np.array(positions)))
    rig.tick(0.1)
    assert all(np.array_equal(asked, RAISED) for asked in _asked(rig)[sent:])
    assert 'not applied' in caplog.text


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
    rig.raise_arm()
    rig.vendor.stuck = True
    rig.stop.stopped = True
    rig.tick(4)
    assert rig.states.emitted[-1][1].status == RobotStatus.BUSY
    rig.vendor.stuck = False
    rig.finish(within_s=3)
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


def test_parking_finishes_after_measured_arrival_and_stopping(rig):
    rig.raise_arm()
    rig.finish(within_s=float(np.max(RAISED)) / DEFAULT_TUNING.max_speed_rad_s + 0.5)
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


def test_a_move_that_cannot_reach_its_target_fails_within_its_bound(world, rig):
    rig.rest()
    rig.vendor.stuck = True
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    tuning = yam.MOVE_SETTLE
    pass_s = float(np.max(RAISED)) / tuning.max_speed_rad_s + tuning.settle_timeout_s
    _until_answered(rig, answer, within_s=tuning.attempts * pass_s)
    with pytest.raises(DeviceFault, match='rests') as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, TimeoutError)
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


def test_a_foreground_yam_that_faults_parks_before_the_world_reports_it(monkeypatch):
    vendor = FakeYam()
    driver = yam.Robot(connect=lambda channel, sim, gravity_comp_factor: vendor)
    fk = yam._Kinematics.fk
    armed = []

    def fk_once_failing(kinematics, q):
        if armed == [True]:
            armed.append(False)
            raise ValueError('forward kinematics failed')
        return fk(kinematics, q)

    monkeypatch.setattr(yam._Kinematics, 'fk', fk_once_failing)
    with pytest.raises(ValueError, match='forward kinematics failed'):
        with pimm.World(virtual_time=True) as world:
            loop = world.start(driver)
            for _ in range(1000):  # past the start move, into the command loop
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

    monkeypatch.setattr('i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface', open_interface)
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
    monkeypatch.setattr(
        'i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface', lambda **kwargs: opened.append(kwargs)
    )
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
    parking_rig.tick(0.5)
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
    with pytest.raises(DeviceFault, match='CAN read failed') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.ARM
    assert isinstance(raised.value.__cause__, yam.ChainFault)
    assert raised.value.__cause__.__cause__ is error
    assert not parking_rig.vendor.released_at
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
    with pytest.raises(DeviceFault, match='CAN write failed') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.ARM
    assert isinstance(raised.value.__cause__, yam.ChainFault)
    assert raised.value.__cause__.__cause__ is error
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


CHAIN_SITES = [Site.CHAIN_READ, Site.CHAIN_COMMAND]
ONCE_PER_CHAIN_SITES = [Site.INFO_READ]  # read once, as the chain opens


@pytest.mark.parametrize('site', [site for site in Site if site not in CHAIN_SITES + ONCE_PER_CHAIN_SITES])
def test_a_fault_at_a_site_off_the_chain_parks_releases_and_is_raised_again(site):
    watch = Watch(site)
    rig = Rig(watch=watch)
    if site not in (Site.META_EMIT, Site.ARM_META_EMIT):  # the metadata goes out once, as the run starts
        rig.raise_arm()
    watch.armed = True
    rig.tick()  # the fault lands in the run, before anything asks the driver to stop
    assert watch.seen >= {site.value} and not watch.armed
    with pytest.raises(OSError, match='collaborator failed') as raised:
        rig.finish(within_s=30)
    assert raised.value is watch.error
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


def _last(emitter: RecordingEmitter):
    return emitter.emitted[-1][1]


def _readier(world, rig: Rig) -> pimm.calls.Caller[None, None]:
    caller = pimm.calls.ControlSystemCaller[None, None](rig.driver)
    wire_call(world, caller, rig.driver.ready)
    return caller


def _lose_the_chain(rig: Rig) -> FakeYam:
    """Fail one chain read on a raised arm, and run past the park. Returns the chain the next open gets."""
    chain = rig.vendor
    read = chain.get_observations
    failures = [OSError('CAN read failed')]

    def read_once_failing():
        if failures:
            raise failures.pop()
        return read()

    rig.raise_arm()
    chain.get_observations = read_once_failing
    rig.tick(10)
    reopened = FakeYam()
    rig.driver._connect = lambda channel, sim, gravity_comp_factor: reopened
    return reopened


@pytest.mark.parametrize('site', CHAIN_SITES)
def test_a_chain_fault_parks_releases_and_puts_an_error_on_the_state_and_the_grip(site):
    watch = Watch(site)
    rig = Rig(watch=watch)
    rig.raise_arm()
    watch.armed = True
    rig.tick(10)

    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed
    for emitter in (rig.states, rig.grips):
        error = _last(emitter)
        assert isinstance(error, DeviceFault) and error.kind is FaultKind.ARM
        assert 'The YAM arm is lost: the chain failed: collaborator failed' in str(_last(emitter))
    rig.finish()


def test_a_chain_whose_robot_info_cannot_be_read_parks_releases_and_holds_the_arm_as_lost():
    watch = Watch(Site.INFO_READ)
    watch.armed = True
    rig = Rig(watch=watch)
    rig.tick(10)

    assert not watch.armed
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed
    for emitter in (rig.states, rig.grips):
        error = _last(emitter)
        assert isinstance(error, DeviceFault) and error.kind is FaultKind.ARM
        assert 'The YAM arm is lost: the chain failed: collaborator failed' in str(error)
    rig.finish()


def _fail_the_first_move(chain: FakeYam) -> None:
    """Fail the first command that moves the chain off the target it holds now, once."""
    send = chain.command_joint_pos
    held = chain.targets[-1] if chain.targets else None
    failures = [OSError('CAN write failed')]

    def command_once_failing(joint_pos):
        if failures and (held is None or not np.array_equal(joint_pos, held)):
            raise failures.pop()
        send(joint_pos)

    chain.command_joint_pos = command_once_failing


def _assert_held_as_a_lost_arm(rig: Rig, chain: FakeYam) -> None:
    assert chain.released_at and chain.closed
    for emitter in (rig.states, rig.grips):
        error = _last(emitter)
        assert isinstance(error, DeviceFault) and error.kind is FaultKind.ARM
        assert 'CAN write failed' in str(error)


def test_a_chain_fault_during_a_blocking_move_answers_the_move_and_holds_the_arm_as_lost(world, rig):
    rig.raise_arm()
    _fail_the_first_move(rig.vendor)

    answer = _sync_caller(world, rig)(command.JointPosition(PARK))
    _until_answered(rig, answer)
    rig.tick(10)

    with pytest.raises(DeviceFault, match='CAN write failed') as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, yam.ChainFault)
    _assert_held_as_a_lost_arm(rig, rig.vendor)


def test_a_chain_fault_during_the_start_move_after_a_ready_call_answers_it_and_holds_the_arm_as_lost(world, rig):
    reopened = _lose_the_chain(rig)
    _fail_the_first_move(reopened)

    answer = _readier(world, rig)(None)
    rig.tick(10)

    with pytest.raises(DeviceFault, match='CAN write failed') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.ARM
    _assert_held_as_a_lost_arm(rig, reopened)


def test_a_ready_call_on_a_lost_arm_is_answered_once_the_start_move_ends(world, rig):
    reopened = _lose_the_chain(rig)

    answer = _readier(world, rig)(None)
    rig.tick(0.1)
    assert not answer.done(), 'the ready call was answered before the arm reached the start pose'
    rig.ready()

    assert answer.result() is None
    np.testing.assert_allclose(reopened._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)


def test_a_chain_fault_during_an_idle_park_holds_the_arm_as_lost(rig):
    rig.raise_arm()
    _fail_the_first_move(rig.vendor)

    rig.tick(10)

    _assert_held_as_a_lost_arm(rig, rig.vendor)


def test_a_move_on_a_lost_arm_is_answered_with_its_error(world, rig):
    _lose_the_chain(rig)

    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    rig.tick()

    with pytest.raises(DeviceFault, match='The YAM arm is lost') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.ARM


def test_a_ready_call_on_a_lost_arm_opens_the_chain_again_and_moves_it_to_the_start_pose(world, rig):
    reopened = _lose_the_chain(rig)

    answer = _readier(world, rig)(None)
    rig.tick(0.1)
    rig.ready()

    assert answer.result() is None
    np.testing.assert_allclose(reopened._pos[:6], RAISED, atol=0.005)
    rig.commands.push(command.JointPosition(PARK))
    rig.tick(1)
    np.testing.assert_allclose(reopened._pos[:6], PARK, atol=0.005)


def test_a_ready_call_whose_chain_does_not_open_answers_its_error_and_the_next_one_opens_it(world, rig):
    _lose_the_chain(rig)
    reopened = FakeYam()

    def refuse(channel, sim, gravity_comp_factor):
        raise OSError('CAN interface can0 is down')

    rig.driver._connect = refuse
    ready = _readier(world, rig)
    first = ready(None)
    rig.tick()

    with pytest.raises(DeviceFault, match='can0 is down') as answered:
        first.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, yam.ChainFault)
    assert isinstance(_last(rig.states), pimm.SignalError)

    rig.driver._connect = lambda channel, sim, gravity_comp_factor: reopened
    second = ready(None)
    rig.tick(5)
    assert second.result() is None


def test_a_ready_call_whose_reopened_arm_misses_its_start_pose_answers_the_miss(world, rig):
    _lose_the_chain(rig)
    reopened = FakeYam()
    reopened.stuck = True
    rig.driver._connect = lambda channel, sim, gravity_comp_factor: reopened
    answer = _readier(world, rig)(None)
    _until_answered(rig, answer, within_s=120)

    with pytest.raises(DeviceFault) as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, TimeoutError)


def test_a_chain_that_does_not_open_at_start_holds_an_error_until_a_ready_call_opens_it(world):
    rig = Rig()
    reopened = FakeYam()
    attempts = []

    def connect(channel, sim, gravity_comp_factor):
        attempts.append(channel)
        if len(attempts) == 1:
            raise OSError('CAN interface can0 is down')
        return reopened

    rig.driver._connect = connect
    rig.tick()

    assert 'can0 is down' in str(_last(rig.states))
    answer = _readier(world, rig)(None)
    rig.tick(0.1)
    rig.ready()
    assert answer.result() is None
    np.testing.assert_allclose(reopened._pos[:6], RAISED, atol=0.005)
    rig.loop.close()


def test_a_ready_call_on_an_open_chain_is_answered_at_once(world, rig):
    rig.ready()
    connected = len(rig.vendor.targets)

    answer = _readier(world, rig)(None)
    rig.tick()

    assert answer.result() is None
    assert not rig.vendor.released_at
    assert len(rig.vendor.targets) > connected, 'the arm stopped taking commands'


_START_MOVE_LOG = 'Moving the arm to the start pose'


def test_a_ready_call_after_a_missed_start_moves_the_arm_to_the_start_pose_again(world, rig, caplog):
    caplog.set_level(logging.INFO)
    rig.vendor.stuck = True
    rig.ready()
    ready = _readier(world, rig)

    missed = ready(None)
    _until_answered(rig, missed, within_s=120)
    with pytest.raises(DeviceFault) as answered:
        missed.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, TimeoutError)
    assert caplog.text.count(_START_MOVE_LOG) == 2

    rig.vendor.stuck = False
    reached = ready(None)
    _until_answered(rig, reached, within_s=120)
    assert reached.result() is None
    assert caplog.text.count(_START_MOVE_LOG) == 3
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=yam.MOVE_SETTLE.tolerance_rad)

    after = ready(None)
    rig.tick()
    assert after.result() is None
    assert caplog.text.count(_START_MOVE_LOG) == 3


def test_a_chain_fault_while_a_ready_call_retries_a_missed_start_answers_it_and_holds_the_arm_as_lost(world, rig):
    rig.vendor.stuck = True
    rig.ready()
    rig.vendor.stuck = False
    _fail_the_first_move(rig.vendor)

    answer = _readier(world, rig)(None)
    _until_answered(rig, answer)
    rig.tick(10)

    with pytest.raises(DeviceFault, match='CAN write failed') as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    _assert_held_as_a_lost_arm(rig, rig.vendor)


def test_a_ready_call_after_a_reached_start_does_not_move_the_arm_to_the_start_pose(world, rig, caplog):
    caplog.set_level(logging.INFO)
    rig.ready()
    rig.commands.push(command.JointPosition(PARK))
    rig.tick(0.5)

    answer = _readier(world, rig)(None)
    rig.tick()

    assert answer.result() is None
    assert caplog.text.count(_START_MOVE_LOG) == 1
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_a_move_that_misses_its_target_puts_no_error_on_the_state(world, rig):
    """A miss is an arm that reads ERROR over a chain that answers, and the arm takes the next command."""
    rig.rest()
    rig.vendor.stuck = True
    answer = _sync_caller(world, rig)(command.JointPosition(RAISED))
    _until_answered(rig, answer, within_s=120)

    with pytest.raises(DeviceFault) as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, TimeoutError)
    assert not any(isinstance(data, pimm.SignalError) for _, data in rig.states.emitted + rig.grips.emitted)


def test_a_blocking_move_with_a_malformed_command_answers_its_own_error_and_not_a_device_fault(world, rig):
    """A command the driver cannot turn into a target is the asker's error, and the arm never sees it."""
    rig.rest()
    far = geom.Transform3D(translation=np.array([10.0, 0.0, 0.0]), rotation=geom.Rotation.identity)
    answer = _sync_caller(world, rig)(command.CartesianPosition(pose=far))
    _until_answered(rig, answer, within_s=5)

    with pytest.raises(ValueError, match='out of reach'):
        answer.result()


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


def _gripper(world, rig: Rig) -> pimm.calls.Caller[float, None]:
    caller = pimm.calls.ControlSystemCaller[float, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_grip)
    return caller


def test_a_chain_fault_during_a_grip_move_answers_it_with_an_arm_fault(world, rig):
    rig.raise_arm()
    answer = _gripper(world, rig)(0.8)
    rig.tick()
    assert not answer.done(), 'the grip move ended before the chain failed'
    read = rig.vendor.get_observations
    failures = [OSError('CAN read failed')]

    def read_once_failing():
        if failures:
            raise failures.pop()
        return read()

    rig.vendor.get_observations = read_once_failing
    _until_answered(rig, answer)

    with pytest.raises(DeviceFault, match='CAN read failed') as answered:
        answer.result()
    assert answered.value.kind is FaultKind.ARM
    assert isinstance(answered.value.__cause__, yam.ChainFault)


def test_a_grip_move_with_no_fault_answers_once_the_fingers_arrive(world, rig):
    rig.raise_arm()
    answer = _gripper(world, rig)(0.8)
    _until_answered(rig, answer)

    assert answer.result() is None
    assert rig.vendor._pos[6] == pytest.approx(0.2, abs=0.05)


def _after_the_first_fault(emitter: RecordingEmitter) -> list:
    values = [data for _, data in emitter.emitted]
    first = next(i for i, data in enumerate(values) if isinstance(data, DeviceFault))
    return values[first:]


def test_a_chain_fault_is_published_once_and_the_park_after_it_publishes_no_readings(rig):
    _lose_the_chain(rig)

    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    for emitter in (rig.states, rig.grips):
        after = _after_the_first_fault(emitter)
        assert len(after) == 1, f'{len(after) - 1} values replaced the fault during or after the park'
        assert isinstance(after[0], DeviceFault) and after[0].kind is FaultKind.ARM


def test_a_shutdown_with_no_fault_publishes_the_parked_readings(rig):
    rig.raise_arm()
    published = len(rig.states.emitted)

    rig.finish()

    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert len(rig.states.emitted) > published
    parked = _last(rig.states)
    assert isinstance(parked, yam.YamState) and parked.status == RobotStatus.AVAILABLE
    np.testing.assert_allclose(parked.q, PARK, atol=0.005)
    assert not any(isinstance(data, pimm.SignalError) for _, data in rig.states.emitted + rig.grips.emitted)


@pytest.mark.parametrize('site', CHAIN_SITES)
def test_a_chain_outage_that_blocks_the_shutdown_leaves_the_error_on_the_state_and_the_grip(site):
    watch = Watch(site, persistent=True)
    rig = Rig(watch=watch)
    rig.raise_arm()
    watch.armed = True
    rig.tick(30)

    assert not rig.vendor.released_at
    for emitter in (rig.states, rig.grips):
        error = _last(emitter)
        assert isinstance(error, DeviceFault) and error.kind is FaultKind.ARM
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
    """The fake chain each driver opens, keyed by its CAN channel. As i2rt does, it reports the gravity compensation
    that the driver passed, or i2rt's own if the driver passed none, and 1.0 for the gripper motor."""
    opened: dict[str, yam._FakeYam] = {}

    def get_yam_robot(channel, gravity_comp_factor, **_):
        chain = opened.setdefault(channel, yam._FakeYam())
        arm = I2RT_GRAVITY_COMP if gravity_comp_factor is None else gravity_comp_factor
        chain.gravity_comp_factor = np.append(arm, 1.0)
        return chain

    monkeypatch.setattr('i2rt.robots.get_robot.get_yam_robot', get_yam_robot)
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
    rig = yam_eval.bimanual.override(embodiment=embodiment_cfg.yam_bimanual.override(cameras={})).instantiate()
    (trial,) = rig.tasks()

    grips = _a_trial_after_a_closed_trial(rig.embodiment, trial.prepare_args, [f'.{s}' for s in keys.BIMANUAL_ARMS])

    assert all(g is not None and g < _GRIP_TOL for g in grips), grips
    assert len(chains) == 2
    for chain in chains.values():
        assert chain.last_command is not None and chain.last_command[6] == pytest.approx(1.0)  # the chain's 1 is open


def test_a_single_arm_trial_starts_with_the_gripper_open_after_a_trial_that_closed_it(chains):
    rig = embodiment_cfg.yam.instantiate()
    start = {eval_keys.ARM: command.JointPosition(np.asarray(roboarm_cfg.YAM_NOMINAL_JOINTS)), eval_keys.GRIPPER: OPEN}

    (grip,) = _a_trial_after_a_closed_trial(rig, start, [''])

    assert grip is not None and grip < _GRIP_TOL, grip
    (chain,) = chains.values()
    assert chain.last_command is not None and chain.last_command[6] == pytest.approx(1.0)


class _Idle(Policy):
    """Commands nothing, so an episode runs until the test ends it."""

    def __init__(self):
        self.observed = False

    def run(self, runtime) -> PolicyRun:
        yield
        while True:
            self.observed = True
            yield Step({}, runtime.time_ns + 100_000_000)


def _record_an_episode(rig: Embodiment, root: Path) -> dict[str, Any]:
    """Run one episode of ``_Idle`` on ``rig``, record it under ``root``, and return the statics it recorded."""
    harness, policy = Harness(rig), _Idle()
    with pimm.World(virtual_time=True) as world:
        wire.wire_embodiment(world, harness, rig)
        perform = world.pair(harness.perform_task)
        done = world.pair(harness.done)
        loop = world.start([harness, *rig.control_systems])
        answer = perform(Rollout(Task('idle', None), policy, LocalDatasetWriter(root)))
        _run_until(loop, lambda: policy.observed)
        done.emit({eval_keys.SUCCESS: True})
        _run_until(loop, answer.done)
        assert answer.result()[eval_keys.TERMINATED]
    return LocalDataset(root)[0].static


@pytest.mark.parametrize(
    'rig, recorded',
    [
        (
            embodiment_cfg.yam_bimanual_yambox.override(cameras={}),
            {
                f'{keys.ROBOT_STATE}.{side}{keys.JOINTS_SUFFIX}': [*STATION_GRAVITY_COMP, 1.0]
                for side in keys.BIMANUAL_ARMS
            },
        ),
        (embodiment_cfg.yam, {keys.JOINTS: [*I2RT_GRAVITY_COMP, 1.0]}),
    ],
    ids=['yambox', 'one-arm'],
)
def test_an_episode_records_the_gravity_compensation_that_each_chain_runs(chains, tmp_path, rig, recorded):
    assert _record_an_episode(rig.instantiate(), tmp_path)[roboarm_keys.GRAVITY_COMP_FACTOR] == recorded


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


def test_the_arm_meta_carries_the_gravity_compensation_that_the_chain_reports(rig):
    rig.vendor.gravity_comp_factor = np.array([*STATION_GRAVITY_COMP, 1.0])
    rig.ready()
    assert _last(rig.arm_metas) == {roboarm_keys.GRAVITY_COMP_FACTOR: [*STATION_GRAVITY_COMP, 1.0]}


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

    monkeypatch.setattr('i2rt.robots.get_robot.get_yam_robot', get_yam_robot)
    with pimm.World() as world:
        loop = world.start(list(rig.override(cameras={}).instantiate().control_systems))
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


def test_the_yam_package_hands_out_the_driver():
    assert yam_package.Robot is yam.Robot


def test_the_settle_tuning_imports_without_the_driver():
    loaded = subprocess.run(
        [sys.executable, '-c', 'import sys, positronic.drivers.roboarm.yam.settle; print(sorted(sys.modules))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert 'positronic.drivers.roboarm.yam.driver' not in loaded
