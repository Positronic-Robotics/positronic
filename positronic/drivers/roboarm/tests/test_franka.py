"""Tests for the Franka arm driver.

The driver imports `positronic_franka`, which ships only in the Linux-only `hardware` extra, so these
tests would not run anywhere the extra is absent — CI included. The vendor package is stood in when it
is genuinely missing, and a real install is left alone so a hardware box still tests against it.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

import pimm

if importlib.util.find_spec('positronic_franka') is None:
    _vendor = types.ModuleType('positronic_franka')
    _vendor_ext = types.ModuleType('positronic_franka._franka')
    _vendor_desk = types.ModuleType('positronic_franka.desk')
    vars(_vendor_ext).update(
        Robot=MagicMock(), RealtimeConfig=MagicMock(), InternalImpedance=MagicMock(), State=MagicMock()
    )
    vars(_vendor_desk).update(Desk=MagicMock(), SafetyControllerError=type('SafetyControllerError', (Exception,), {}))
    vars(_vendor).update(_franka=_vendor_ext, desk=_vendor_desk)
    sys.modules['positronic_franka'] = _vendor
    sys.modules['positronic_franka._franka'] = _vendor_ext
    sys.modules['positronic_franka.desk'] = _vendor_desk

from positronic.drivers.roboarm import command, franka  # noqa: E402  # needs the vendor stand-in above

HOME_JOINTS = [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
FR3_URDF = (Path(__file__).resolve().parent.parent / 'fr3.urdf').read_text()


class _FrankaState:
    """One `pf.State` snapshot: what the arm reports for a single control tick."""

    def __init__(self, q, error, error_message):
        self.q = q
        self.q_d = q
        self.dq = np.zeros(7)
        self.tau_J = np.zeros(7)
        self.tau_J_d = np.zeros(7)
        self.end_effector_pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.ee_wrench = np.zeros(6)
        self.error = error
        self.error_message = error_message
        self.time = 0.0


class _FakeFranka:
    """Stand-in for `pf.Robot`, with a reflex the test fires on demand.

    An armed reflex trips on the next commanded move and leaves the arm reporting the error, which is what the
    real arm does. A blocking move additionally raises, because the control thread it is waiting on has stopped.
    """

    def __init__(self, q):
        self.q = np.asarray(q, dtype=np.float64)
        self.error = 0
        self.error_message = '[]'
        self.recover_calls = 0
        self.moves: list[tuple[np.ndarray, bool]] = []
        self._reflex_armed = False
        self.fail_with = None

    def arm_reflex(self):
        self._reflex_armed = True

    def state(self):
        return _FrankaState(self.q, self.error, self.error_message)

    def set_target_joints(self, q_target, asynchronous=True):
        self.moves.append((np.asarray(q_target, dtype=np.float64), asynchronous))
        if self.fail_with is not None:
            raise self.fail_with
        if self._reflex_armed:
            self._reflex_armed = False
            self.error = 1
            self.error_message = '["cartesian_reflex"]'
            if not asynchronous:
                raise RuntimeError('control loop stopped before the joint target was reached')
            return
        self.q = np.asarray(q_target, dtype=np.float64)

    def recover_from_errors(self):
        self.recover_calls += 1
        self.error = 0
        self.error_message = '[]'
        return True

    def stop(self):
        pass

    def get_robot_model(self):
        return FR3_URDF

    def set_collision_behavior(self, **kwargs):
        pass

    def set_control_mode(self, mode):
        pass

    def set_load(self, *args):
        pass


@pytest.fixture
def arm(monkeypatch):
    """A Franka driver wired to a fake arm, ready to be driven through a `pimm.World`."""
    fake = _FakeFranka(HOME_JOINTS)
    monkeypatch.setattr(franka.pf, 'Robot', lambda *args, **kwargs: fake)
    return fake


def _bind_input(world, receiver) -> pimm.SignalEmitter:
    """The emitter side of a driver input. Cast because a driver declares its endpoints as the base signal
    classes, while `World.pair` is typed against the control-system ones it actually receives."""
    return cast(pimm.SignalEmitter, world.pair(cast(pimm.ControlSystemReceiver, receiver)))


def _bind_output(world, emitter) -> None:
    """Give a driver output somewhere to emit to, so nothing is dropped on an unbound endpoint."""
    world.pair(cast(pimm.ControlSystemEmitter, emitter))


def _driver():
    """The arm driver with Desk left to the operator and homing variation off, so the target is exact."""
    return franka.Robot('0.0.0.0', manage_desk=False, home_joints_variation=[0.0] * 7)


def test_reflex_while_homing_is_recovered(arm):
    """Homing on a Reset command survives a reflex, the way the control loop survives every other reflex."""
    with pimm.World(virtual_time=True) as world:
        robot = _driver()
        commands = _bind_input(world, robot.commands)
        _bind_output(world, robot.state)
        _bind_output(world, robot.robot_meta)

        loop = world.start([robot])
        for _ in range(3):
            next(loop)
        recovered_before, moves_before = arm.recover_calls, len(arm.moves)

        commands.emit([(world.clock.now_ns(), command.Reset())])
        arm.arm_reflex()
        for _ in range(30):
            next(loop)

        assert len(arm.moves) > moves_before, 'the Reset command never reached the driver'
        assert arm.recover_calls > recovered_before, 'the reflex was never cleared'
        assert arm.error == 0


def test_reflex_while_homing_leaves_the_driver_serving_commands(arm):
    """The point of clearing the reflex: the driver is still there afterwards to take the next command."""
    with pimm.World(virtual_time=True) as world:
        robot = _driver()
        commands = _bind_input(world, robot.commands)
        _bind_output(world, robot.state)
        _bind_output(world, robot.robot_meta)

        loop = world.start([robot])
        for _ in range(3):
            next(loop)

        commands.emit([(world.clock.now_ns(), command.Reset())])
        arm.arm_reflex()
        for _ in range(30):
            next(loop)
        moves_after_reflex = len(arm.moves)

        commands.emit([(world.clock.now_ns(), command.JointPosition(positions=np.array(HOME_JOINTS)))])
        for _ in range(30):
            next(loop)

        assert len(arm.moves) > moves_after_reflex, 'the driver stopped serving commands after the reflex'


def test_reflex_during_the_startup_home_is_recovered(arm):
    """The driver homes on start too, and that call blocks on the same control thread."""
    arm.arm_reflex()
    with pimm.World(virtual_time=True) as world:
        robot = _driver()
        _bind_output(world, robot.state)
        _bind_output(world, robot.robot_meta)

        loop = world.start([robot])
        for _ in range(5):
            next(loop)

        assert arm.recover_calls > 1, 'the reflex during the startup home was never cleared'
        assert arm.error == 0


def test_homing_failure_the_arm_does_not_report_is_not_swallowed(arm):
    """Only a fault the arm reports is ours to clear; anything else still takes the driver down."""
    arm.fail_with = RuntimeError('libfranka: connection closed by robot')
    with pimm.World(virtual_time=True) as world:
        robot = _driver()
        _bind_output(world, robot.state)
        _bind_output(world, robot.robot_meta)

        loop = world.start([robot])
        with pytest.raises(RuntimeError, match='connection closed by robot'):
            for _ in range(5):
                next(loop)
