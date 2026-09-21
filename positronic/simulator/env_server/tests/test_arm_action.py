"""One arm's wire command as absolute joint targets, checked without any benchmark runtime."""

import numpy as np
import pytest

from positronic.simulator.env_server import arm_action, protocol

_JOINTS = 7
_MEASURED_EEF = (np.zeros(3), np.eye(3))


def _action(command, current, *, ik=lambda _pos, _rot: np.zeros(_JOINTS), current_eef=_MEASURED_EEF):
    return arm_action.wire_command_to_arm_action(command, current, ik=ik, current_eef=current_eef)


def test_wire_command_joint_pos_passthrough():
    current = np.arange(_JOINTS, dtype=np.float32)
    q = np.full(_JOINTS, 0.3, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: q}, current)
    assert out.dtype == np.float32 and out.shape == (_JOINTS,)
    assert np.array_equal(out, q)


def test_wire_command_joint_delta_adds_to_measured():
    current = np.arange(_JOINTS, dtype=np.float32)
    dq = np.full(_JOINTS, 0.1, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: dq}, current)
    assert np.allclose(out, current + dq)


def test_wire_command_hold_recommands_measured():
    current = np.linspace(-1.0, 1.0, _JOINTS, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.HOLD}, current)
    assert np.array_equal(out, current)


def test_wire_command_joint_count_mismatch_raises():
    current = np.zeros(_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        _action(
            {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: np.zeros(6, dtype=np.float32)},
            current,
        )


def test_unpack_wire_pose_round_trips_translation_and_rotation():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    pos, out = arm_action.unpack_wire_pose(np.concatenate([[1.0, 2.0, 3.0], rot.reshape(-1)]))
    assert np.array_equal(pos, [1.0, 2.0, 3.0])
    assert np.array_equal(out, rot)


def test_unpack_wire_pose_rejects_wrong_width():
    with pytest.raises(ValueError):
        arm_action.unpack_wire_pose(np.zeros(7))


def test_compose_world_delta_adds_translation_and_left_multiplies_rotation():
    cur_rot = np.eye(3)
    delta_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos, rot = arm_action.compose_world_delta([1.0, 0.0, 0.0], cur_rot, [0.0, 2.0, 0.0], delta_rot)
    assert np.allclose(pos, [1.0, 2.0, 0.0])
    assert np.allclose(rot, delta_rot)


def test_cartesian_command_resolves_through_the_supplied_ik():
    solved = np.arange(_JOINTS, dtype=np.float64)
    seen = {}

    def ik(pos, rot):
        seen['pos'], seen['rot'] = pos, rot
        return solved

    rot = np.eye(3)
    cmd = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN,
        protocol.COMMAND_POSE: np.concatenate([[0.4, 0.1, 0.3], rot.reshape(-1)]),
    }
    out = _action(cmd, np.zeros(_JOINTS), ik=ik)
    assert out.dtype == np.float32 and np.allclose(out, solved)
    assert np.allclose(seen['pos'], [0.4, 0.1, 0.3]) and np.allclose(seen['rot'], rot)


def test_cartesian_delta_composes_onto_the_measured_eef_before_solving():
    seen = {}

    def ik(pos, rot):
        seen['pos'], seen['rot'] = pos, rot
        return np.zeros(_JOINTS)

    cmd = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA,
        protocol.COMMAND_DELTA: np.concatenate([[0.0, 0.1, 0.0], np.eye(3).reshape(-1)]),
    }
    _action(cmd, np.zeros(_JOINTS), ik=ik, current_eef=(np.array([0.5, 0.0, 0.2]), np.eye(3)))
    assert np.allclose(seen['pos'], [0.5, 0.1, 0.2])


def test_unknown_command_names_the_canonical_contract():
    with pytest.raises(ValueError, match='cartesian'):
        _action({protocol.COMMAND_TYPE: 'wrench'}, np.zeros(_JOINTS))


@pytest.mark.parametrize('command_type', protocol.CANONICAL_COMMAND_TYPES)
def test_every_canonical_command_type_converts_to_joint_targets(command_type):
    pose = np.concatenate([np.zeros(3), np.eye(3).reshape(-1)])
    payload = {
        protocol.JOINT_POS: {protocol.COMMAND_JOINT_POS: np.zeros(_JOINTS)},
        protocol.JOINT_DELTA: {protocol.COMMAND_JOINT_DELTA: np.zeros(_JOINTS)},
        protocol.HOLD: {},
        protocol.CARTESIAN: {protocol.COMMAND_POSE: pose},
        protocol.CARTESIAN_DELTA: {protocol.COMMAND_DELTA: pose},
    }[command_type]

    target = _action(
        {protocol.COMMAND_TYPE: command_type, **payload},
        np.zeros(_JOINTS),
        ik=lambda _pos, _rot: np.zeros(_JOINTS),
        current_eef=(np.zeros(3), np.eye(3)),
    )

    assert target.shape == (_JOINTS,)
