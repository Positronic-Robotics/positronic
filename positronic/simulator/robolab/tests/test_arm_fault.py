"""The fault of a RoboLab clone's arm state, checked without Isaac Lab on the Panda joint limits."""

import numpy as np
import pytest

from positronic.simulator.robolab import keys
from positronic.simulator.robolab.arm_fault import arm_faults

_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_RANGE = _UPPER - _LOWER
_HOME = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0])
_CONTROL_DT = 1 / 15


def _faults(joint_pos, joint_vel=None) -> list[str | None]:
    joint_pos = np.atleast_2d(joint_pos)
    joint_vel = np.zeros_like(joint_pos) if joint_vel is None else np.atleast_2d(joint_vel)
    limits = np.broadcast_to(np.stack([_LOWER, _UPPER], axis=-1), (*joint_pos.shape, 2))
    return arm_faults(joint_pos, joint_vel, limits, _CONTROL_DT)


def _with(values: np.ndarray, joint: int, value: float) -> np.ndarray:
    values = values.copy()
    values[joint] = value
    return values


def test_an_arm_inside_its_limits_has_no_fault():
    assert _faults(_HOME) == [None]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_a_joint_or_joint_velocity_that_is_not_finite_is_a_nonfinite_fault(value):
    assert _faults(_with(_HOME, 6, value)) == [keys.FAULT_NONFINITE_ARM]
    assert _faults(_HOME, _with(np.zeros(7), 6, value)) == [keys.FAULT_NONFINITE_ARM]


def test_a_finite_joint_far_past_its_limit_is_an_out_of_range_fault():
    assert _faults(_with(_HOME, 6, -37.5)) == [keys.FAULT_ARM_OUT_OF_RANGE]


def test_a_finite_joint_speed_far_past_any_arm_speed_is_an_out_of_range_fault():
    assert _faults(_HOME, _with(np.zeros(7), 4, 141.0)) == [keys.FAULT_ARM_OUT_OF_RANGE]


def test_contact_overshoot_at_a_limit_and_fast_chatter_have_no_fault():
    # A contact at a limit overshoots it by about 0.13 rad, and a chattering arm moves at about 10 rad/s.
    overshoot = _with(_HOME, 0, _LOWER[0] - 0.13)
    assert _faults(overshoot, _with(np.zeros(7), 0, 6.1)) == [None]
    assert _faults(_HOME, _with(np.zeros(7), 6, 10.2)) == [None]


@pytest.mark.parametrize('joint', range(7))
def test_a_joint_faults_only_past_one_joint_range_beyond_a_limit(joint):
    lower, upper, span = float(_LOWER[joint]), float(_UPPER[joint]), float(_RANGE[joint])
    assert _faults(_with(_HOME, joint, lower - 0.999 * span)) == [None]
    assert _faults(_with(_HOME, joint, upper + 0.999 * span)) == [None]
    assert _faults(_with(_HOME, joint, lower - 1.001 * span)) == [keys.FAULT_ARM_OUT_OF_RANGE]
    assert _faults(_with(_HOME, joint, upper + 1.001 * span)) == [keys.FAULT_ARM_OUT_OF_RANGE]


@pytest.mark.parametrize('joint', range(7))
def test_a_joint_speed_faults_only_past_one_joint_range_per_control_step(joint):
    limit = float(_RANGE[joint]) / _CONTROL_DT
    assert _faults(_HOME, _with(np.zeros(7), joint, -0.999 * limit)) == [None]
    assert _faults(_HOME, _with(np.zeros(7), joint, -1.001 * limit)) == [keys.FAULT_ARM_OUT_OF_RANGE]


def test_a_state_both_not_finite_and_out_of_range_is_a_nonfinite_fault():
    state = _with(_with(_HOME, 6, -37.5), 0, float('nan'))
    assert _faults(state) == [keys.FAULT_NONFINITE_ARM]


def test_each_slot_has_its_own_fault():
    joint_pos = np.stack([_HOME, _with(_HOME, 6, -37.5), _HOME, _with(_HOME, 2, float('nan'))])
    assert _faults(joint_pos) == [None, keys.FAULT_ARM_OUT_OF_RANGE, None, keys.FAULT_NONFINITE_ARM]
