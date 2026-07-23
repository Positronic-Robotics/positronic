"""Unit tests for the pure MolmoSpaces <-> wire mappings.

Runs with NEITHER molmo_spaces nor positronic's heavy stack: ``mapping`` imports only numpy, so these pin the
gripper normalization, the wire-command -> joint-target integration, and the camera-key precedence without a
sim or a GPU.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_mapping.py --no-cov
"""

import numpy as np
import pytest

from positronic.simulator.molmo_spaces import mapping


def test_grip_qpos_normalization():
    closed = mapping.GRIPPER_QPOS_CLOSED
    assert mapping.normalize_grip_qpos(0.0) == 0.0
    assert abs(mapping.normalize_grip_qpos(closed / 2) - 0.5) < 1e-6
    assert abs(mapping.normalize_grip_qpos(closed) - 1.0) < 1e-6
    assert mapping.normalize_grip_qpos(closed * 2) == 1.0  # saturates, never exceeds 1
    # A two-finger qpos reads the first finger.
    assert abs(mapping.normalize_grip_qpos(np.array([closed / 2, closed / 2])) - 0.5) < 1e-6


def test_grip_command_to_actuator():
    assert mapping.grip_command_to_actuator(0.0) == mapping.ROBOTIQ_OPEN == 0.0
    assert mapping.grip_command_to_actuator(1.0) == mapping.ROBOTIQ_CLOSED == 255.0
    assert mapping.grip_command_to_actuator(0.5) == 127.5  # continuous — the codec owns binarization
    assert mapping.grip_command_to_actuator(2.0) == 255.0  # clipped


def test_wire_command_joint_pos_passthrough():
    current = np.arange(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    q = np.full(mapping.NUM_ARM_JOINTS, 0.3, dtype=np.float32)
    out = mapping.wire_command_to_arm_action({'type': 'joint_pos', 'q': q}, current)
    assert out.dtype == np.float32 and out.shape == (mapping.NUM_ARM_JOINTS,)
    assert np.array_equal(out, q)  # absolute target, independent of the measured joints


def test_wire_command_joint_vel_integrates_onto_measured():
    current = np.arange(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    dq = np.full(mapping.NUM_ARM_JOINTS, 0.1, dtype=np.float32)
    out = mapping.wire_command_to_arm_action({'type': 'joint_vel', 'dq': dq}, current)
    assert np.allclose(out, current + dq)  # positronic applies JointDelta as q + dq


def test_wire_command_hold_recommands_measured():
    current = np.linspace(-1.0, 1.0, mapping.NUM_ARM_JOINTS, dtype=np.float32)
    out = mapping.wire_command_to_arm_action({'type': 'hold'}, current)
    assert np.array_equal(out, current)


def test_wire_command_joint_count_mismatch_raises():
    current = np.zeros(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        mapping.wire_command_to_arm_action({'type': 'joint_vel', 'dq': np.zeros(6, dtype=np.float32)}, current)


def test_wire_command_cartesian_unsupported():
    current = np.zeros(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        mapping.wire_command_to_arm_action({'type': 'cartesian', 'pose': np.zeros(12)}, current)


def test_camera_key_default_and_variant_precedence():
    default = mapping.MOLMO_WRIST_CAMERA
    variants = mapping.MOLMO_WRIST_CAMERA_VARIANTS
    # Default present, no variant -> the default.
    assert mapping.resolve_camera_key({default: 1}, default, default, variants) == default
    # A benchmark-variant key present wins over the default (matches molmo_spaces pi_policy precedence).
    both = {default: 1, variants[0]: 1}
    assert mapping.resolve_camera_key(both, default, default, variants) == variants[0]
    # Variant only (default absent) -> the variant.
    assert mapping.resolve_camera_key({variants[0]: 1}, default, default, variants) == variants[0]


def test_camera_key_explicit_nondefault_read_as_is():
    # An explicitly configured non-default key is read as-is, never shadowed by a variant decoy.
    obs = {'my_cam': 1, mapping.MOLMO_WRIST_CAMERA_VARIANTS[0]: 1}
    assert mapping.resolve_camera_key(obs, 'my_cam', mapping.MOLMO_WRIST_CAMERA, ()) == 'my_cam'


def test_camera_key_miss_raises():
    with pytest.raises(KeyError):
        mapping.resolve_camera_key({'other': 1}, mapping.MOLMO_WRIST_CAMERA, mapping.MOLMO_WRIST_CAMERA, ())
