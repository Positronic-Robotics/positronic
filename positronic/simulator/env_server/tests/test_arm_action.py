"""One arm's wire command as absolute joint targets, checked without any benchmark runtime."""

import logging

import numpy as np
import pytest

from positronic.simulator.env_server import arm_action, protocol

_JOINTS = 7


def _measured_eef():
    return np.zeros(3), np.eye(3)


def _action(command, current, *, ik=lambda _pos, _rot: np.zeros(_JOINTS), read_eef=_measured_eef) -> np.ndarray:
    target = arm_action.arm_targets(command, current, ik=ik, read_eef=read_eef)
    assert target.refusal is None
    return target.q


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


def test_unpack_wire_pose_round_trips_translation_and_rotation():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    pos, out = arm_action.unpack_wire_pose(np.concatenate([[1.0, 2.0, 3.0], rot.reshape(-1)]))
    assert np.array_equal(pos, [1.0, 2.0, 3.0])
    assert np.array_equal(out, rot)


def test_unpack_wire_pose_rejects_wrong_width():
    with pytest.raises(ValueError):
        arm_action.unpack_wire_pose(np.zeros(7))


def test_compose_reference_delta_adds_translation_and_left_multiplies_rotation():
    cur_rot = np.eye(3)
    delta_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos, rot = arm_action.compose_reference_delta([1.0, 0.0, 0.0], cur_rot, [0.0, 2.0, 0.0], delta_rot)
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
    _action(cmd, np.zeros(_JOINTS), ik=ik, read_eef=lambda: (np.array([0.5, 0.0, 0.2]), np.eye(3)))
    assert np.allclose(seen['pos'], [0.5, 0.1, 0.2])


def test_unknown_command_names_the_canonical_contract():
    with pytest.raises(ValueError, match='cartesian'):
        _action({protocol.COMMAND_TYPE: 'wrench'}, np.zeros(_JOINTS))


_POSE = np.concatenate([np.zeros(3), np.eye(3).reshape(-1)])


@pytest.mark.parametrize('command_type', protocol.CANONICAL_COMMAND_TYPES)
def test_every_canonical_command_type_converts_to_joint_targets(command_type):
    payload = {
        protocol.JOINT_POS: {protocol.COMMAND_JOINT_POS: np.zeros(_JOINTS)},
        protocol.JOINT_DELTA: {protocol.COMMAND_JOINT_DELTA: np.zeros(_JOINTS)},
        protocol.HOLD: {},
        protocol.CARTESIAN: {protocol.COMMAND_POSE: _POSE},
        protocol.CARTESIAN_DELTA: {protocol.COMMAND_DELTA: _POSE},
    }[command_type]

    target = _action(
        {protocol.COMMAND_TYPE: command_type, **payload},
        np.zeros(_JOINTS),
        ik=lambda _pos, _rot: np.zeros(_JOINTS),
        read_eef=_measured_eef if command_type == protocol.CARTESIAN_DELTA else lambda: pytest.fail('read the eef'),
    )

    assert target.shape == (_JOINTS,)


@pytest.mark.parametrize(
    ('command', 'refusal'),
    [
        (
            {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: np.zeros(_JOINTS + 1)},
            f'width {_JOINTS + 1} for {_JOINTS} joints',
        ),
        (
            {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: np.zeros(1)},
            f'width 1 for {_JOINTS} joints',
        ),
        (
            {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: np.zeros(_JOINTS - 1)},
            f'width {_JOINTS - 1} for {_JOINTS} joints',
        ),
        (
            {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: np.full(_JOINTS, np.nan)},
            'not finite',
        ),
        ({protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: _POSE}, 'no IK solution'),
        ({protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA, protocol.COMMAND_DELTA: _POSE}, 'no IK solution'),
    ],
    ids=[
        'delta too wide',
        'delta too narrow',
        'target too narrow',
        'target not finite',
        'pose with no IK solution',
        'pose delta with no IK solution',
    ],
)
def test_a_command_the_env_refuses_holds_the_measured_joints_with_one_warning_that_names_the_arm(
    command, refusal, caplog
):
    measured = np.arange(_JOINTS, dtype=np.float32)

    target = arm_action.arm_targets(command, measured, ik=lambda *_: None, read_eef=_measured_eef, arm='left')

    np.testing.assert_array_equal(target.q, measured)
    assert refusal in target.refusal
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert (
        record.getMessage()
        == f'{command[protocol.COMMAND_TYPE]} command not applied, the left arm holds: {target.refusal}'
    )


def test_a_command_the_env_accepts_passes_through_with_no_log(caplog):
    q = np.full(_JOINTS, 0.3, dtype=np.float32)
    command = {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: q}

    target = arm_action.arm_targets(command, np.zeros(_JOINTS), ik=lambda *_: None, read_eef=_measured_eef)

    np.testing.assert_array_equal(target.q, q)
    assert target.refusal is None
    assert not caplog.records
