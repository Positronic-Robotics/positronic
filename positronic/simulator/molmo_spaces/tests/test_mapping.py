"""Unit tests for the pure MolmoSpaces <-> wire mappings.

Runs with NEITHER molmo_spaces nor positronic's heavy stack: ``mapping`` imports only numpy, so these pin the
gripper normalization, the wire-command -> joint-target integration, and the camera-key precedence without a
sim or a GPU.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_mapping.py --no-cov
"""

import types

import numpy as np
import pytest

from positronic.simulator.env_server import protocol
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
    out = mapping.wire_command_to_arm_action(
        {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: q}, current
    )
    assert out.dtype == np.float32 and out.shape == (mapping.NUM_ARM_JOINTS,)
    assert np.array_equal(out, q)  # absolute target, independent of the measured joints


def test_wire_command_joint_vel_integrates_onto_measured():
    current = np.arange(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    dq = np.full(mapping.NUM_ARM_JOINTS, 0.1, dtype=np.float32)
    out = mapping.wire_command_to_arm_action(
        {protocol.COMMAND_TYPE: protocol.JOINT_VEL, protocol.COMMAND_JOINT_VEL: dq}, current
    )
    assert np.allclose(out, current + dq)  # positronic applies JointDelta as q + dq


def test_wire_command_hold_recommands_measured():
    current = np.linspace(-1.0, 1.0, mapping.NUM_ARM_JOINTS, dtype=np.float32)
    out = mapping.wire_command_to_arm_action({protocol.COMMAND_TYPE: protocol.HOLD}, current)
    assert np.array_equal(out, current)


def test_wire_command_joint_count_mismatch_raises():
    current = np.zeros(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        mapping.wire_command_to_arm_action(
            {protocol.COMMAND_TYPE: protocol.JOINT_VEL, protocol.COMMAND_JOINT_VEL: np.zeros(6, dtype=np.float32)},
            current,
        )


def test_wire_command_cartesian_unsupported():
    current = np.zeros(mapping.NUM_ARM_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        mapping.wire_command_to_arm_action(
            {protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: np.zeros(12)}, current
        )


def test_camera_key_default_and_variant_precedence():
    default = mapping.MOLMO_WRIST_CAMERA
    variants = mapping.MOLMO_WRIST_CAMERA_VARIANTS
    # Default present, no variant -> the default.
    assert mapping.resolve_camera_key({default: 1}, default, variants) == default
    # A benchmark-variant key present wins over the default (matches molmo_spaces pi_policy precedence).
    both = {default: 1, variants[0]: 1}
    assert mapping.resolve_camera_key(both, default, variants) == variants[0]
    # Variant only (default absent) -> the variant.
    assert mapping.resolve_camera_key({variants[0]: 1}, default, variants) == variants[0]


def test_camera_key_explicit_nondefault_read_as_is():
    # An explicitly configured non-default key is read as-is, never shadowed by a variant decoy.
    obs = {'my_cam': 1, mapping.MOLMO_WRIST_CAMERA_VARIANTS[0]: 1}
    assert mapping.resolve_camera_key(obs, 'my_cam') == 'my_cam'


def test_camera_key_miss_raises():
    with pytest.raises(KeyError):
        mapping.resolve_camera_key({'other': 1}, mapping.MOLMO_WRIST_CAMERA)


def _episodes(*horizons_sec):
    return [
        types.SimpleNamespace(task={} if sec is None else {mapping.MOLMO_TASK_HORIZON_SEC: sec}) for sec in horizons_sec
    ]


def test_task_horizon_reads_the_benchmark_task_dict():
    # Where MolmoSpaces' benchmark generator writes it, and where determine_task_horizon reads it.
    assert mapping.resolve_task_horizon_steps(_episodes(30, 30), 66.0) == 455  # round(30 * 1000 / 66)


def test_task_horizon_missing_raises():
    # The raise mirrors upstream's resolver.
    with pytest.raises(ValueError):
        mapping.resolve_task_horizon_steps(_episodes(30, None), 66.0)


def test_task_horizon_disagreeing_across_episodes_raises():
    # The horizon belongs to the benchmark, so one run has one; upstream refuses the same manifest.
    with pytest.raises(ValueError, match='inconsistent'):
        mapping.resolve_task_horizon_steps(_episodes(20, 30), 66.0)


def test_task_horizon_non_positive_raises():
    # A zero or negative span is no horizon at all; the override path already refuses one below a step.
    with pytest.raises(ValueError, match='non-positive'):
        mapping.resolve_task_horizon_steps(_episodes(0), 66.0)
    with pytest.raises(ValueError, match='non-positive'):
        mapping.resolve_task_horizon_steps(_episodes(-5), 66.0)


def test_task_horizon_rounding_below_one_step_raises():
    # 0.03s of a 66ms period rounds to 0 steps, which would expire the episode before its first action.
    with pytest.raises(ValueError, match='rounds to 0 steps'):
        mapping.resolve_task_horizon_steps(_episodes(0.03), 66.0)


def test_task_horizon_override_wins():
    # An explicit override pins the horizon, beating the benchmark field (mirrors --task_horizon_steps), and lets
    # a benchmark that declares none, or disagrees, still resolve.
    assert mapping.resolve_task_horizon_steps(_episodes(20), 66.0, override_steps=500) == 500
    assert mapping.resolve_task_horizon_steps(_episodes(20, 30, None), 66.0, override_steps=455) == 455


def test_exterior_camera_variants_cover_light_randomization_and_randcam():
    # The default exterior mapping must resolve both benchmark exterior names: RandCam records
    # randomized_zed2_analogue_1, not exo_camera_1.
    default = mapping.MOLMO_EXTERIOR_CAMERA
    variants = mapping.MOLMO_EXTERIOR_CAMERA_VARIANTS
    for name in ('droid_shoulder_light_randomization', 'randomized_zed2_analogue_1'):
        assert mapping.resolve_camera_key({name: 1}, default, variants) == name


def test_unpack_wire_pose_round_trips_translation_and_rotation():
    # The client encodes a pose as Transform3D.as_vector(ROTATION_MATRIX): translation, then R row-major.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    pos, out = mapping.unpack_wire_pose(np.concatenate([[1.0, 2.0, 3.0], rot.reshape(-1)]))
    assert np.array_equal(pos, [1.0, 2.0, 3.0])
    assert np.array_equal(out, rot)


def test_unpack_wire_pose_rejects_wrong_width():
    with pytest.raises(ValueError):
        mapping.unpack_wire_pose(np.zeros(7))  # a quaternion-encoded pose is not the wire form


def test_compose_world_delta_adds_translation_and_left_multiplies_rotation():
    # World-frame convention: goal_pos = ee_pos + dpos, goal_ori = R(delta) @ ee_ori. Left-multiplication is
    # what keeps the delta world-framed; composing in the body frame would rotate the translation too.
    cur_rot = np.eye(3)
    delta_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos, rot = mapping.compose_world_delta([1.0, 0.0, 0.0], cur_rot, [0.0, 2.0, 0.0], delta_rot)
    assert np.allclose(pos, [1.0, 2.0, 0.0])
    assert np.allclose(rot, delta_rot)


def test_cartesian_command_resolves_through_the_supplied_ik():
    # env.py owns the solver (it needs the live model); mapping only routes the target into it.
    solved = np.arange(mapping.NUM_ARM_JOINTS, dtype=np.float64)
    seen = {}

    def ik(pos, rot):
        seen['pos'], seen['rot'] = pos, rot
        return solved

    rot = np.eye(3)
    cmd = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN,
        protocol.COMMAND_POSE: np.concatenate([[0.4, 0.1, 0.3], rot.reshape(-1)]),
    }
    out = mapping.wire_command_to_arm_action(cmd, np.zeros(mapping.NUM_ARM_JOINTS), ik=ik)
    assert out.dtype == np.float32 and np.allclose(out, solved)
    assert np.allclose(seen['pos'], [0.4, 0.1, 0.3]) and np.allclose(seen['rot'], rot)


def test_cartesian_delta_composes_onto_the_measured_eef_before_solving():
    # The delta is relative to the *measured* pose, so the solver must see the composed absolute target.
    seen = {}

    def ik(pos, rot):
        seen['pos'], seen['rot'] = pos, rot
        return np.zeros(mapping.NUM_ARM_JOINTS)

    cmd = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA,
        protocol.COMMAND_DELTA: np.concatenate([[0.0, 0.1, 0.0], np.eye(3).reshape(-1)]),
    }
    mapping.wire_command_to_arm_action(
        cmd, np.zeros(mapping.NUM_ARM_JOINTS), ik=ik, current_eef=(np.array([0.5, 0.0, 0.2]), np.eye(3))
    )
    assert np.allclose(seen['pos'], [0.5, 0.1, 0.2])


def test_cartesian_without_an_ik_solver_raises():
    # A caller that holds no model cannot resolve a Cartesian target — fail loud rather than silently holding.
    cmd = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN,
        protocol.COMMAND_POSE: np.concatenate([np.zeros(3), np.eye(3).reshape(-1)]),
    }
    with pytest.raises(ValueError, match='ik solver'):
        mapping.wire_command_to_arm_action(cmd, np.zeros(mapping.NUM_ARM_JOINTS))


def test_unknown_command_names_the_canonical_contract():
    with pytest.raises(ValueError, match='cartesian'):  # the message lists the contract the tag is not part of
        mapping.wire_command_to_arm_action({protocol.COMMAND_TYPE: 'wrench'}, np.zeros(mapping.NUM_ARM_JOINTS))


@pytest.mark.parametrize('command_type', protocol.CANONICAL_COMMAND_TYPES)
def test_every_canonical_command_type_converts_to_joint_targets(command_type):
    """The contract is total, so every canonical type converts to the joint targets MolmoSpaces natively steps.
    This is the model-free half of that property — the routing, through a stub solver; ``validate.py`` drives
    the same types through the real IK against a live scene."""
    pose = np.concatenate([np.zeros(3), np.eye(3).reshape(-1)])
    payload = {
        protocol.JOINT_POS: {protocol.COMMAND_JOINT_POS: np.zeros(mapping.NUM_ARM_JOINTS)},
        protocol.JOINT_VEL: {protocol.COMMAND_JOINT_VEL: np.zeros(mapping.NUM_ARM_JOINTS)},
        protocol.HOLD: {},
        protocol.CARTESIAN: {protocol.COMMAND_POSE: pose},
        protocol.CARTESIAN_DELTA: {protocol.COMMAND_DELTA: pose},
    }[command_type]

    target = mapping.wire_command_to_arm_action(
        {protocol.COMMAND_TYPE: command_type, **payload},
        np.zeros(mapping.NUM_ARM_JOINTS),
        ik=lambda _pos, _rot: np.zeros(mapping.NUM_ARM_JOINTS),
        current_eef=(np.zeros(3), np.eye(3)),
    )

    assert target.shape == (mapping.NUM_ARM_JOINTS,)


def test_episode_seed_prefers_the_override_then_the_spec_then_the_index():
    spec = types.SimpleNamespace(seed=7)
    assert mapping.resolve_episode_seed(spec, 3, 99) == 99
    assert mapping.resolve_episode_seed(spec, 3) == 7
    assert mapping.resolve_episode_seed(types.SimpleNamespace(seed=None), 3) == 3
    assert mapping.resolve_episode_seed(types.SimpleNamespace(), 5) == 5
