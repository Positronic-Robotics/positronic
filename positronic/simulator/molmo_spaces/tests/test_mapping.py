"""Unit tests for the pure MolmoSpaces <-> wire mappings.

Runs with NEITHER molmo_spaces nor positronic's heavy stack: ``mapping`` imports only numpy, so these pin the
gripper normalization, the wire-command -> joint-target integration, and the camera-key precedence without a
sim or a GPU.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_mapping.py --no-cov
"""

import types
from pathlib import Path

import numpy as np
import pytest

from positronic.simulator.env_server import protocol
from positronic.simulator.molmo_spaces import mapping

# The DROID rig runs 7 Franka arm joints. ``ik`` and the measured pose come from ``env.py`` in production;
# here they are stubs, so these tests pin the routing rather than the kinematics.
_JOINTS = 7
_MEASURED_EEF = (np.zeros(3), np.eye(3))


def _action(command, current, *, ik=lambda _pos, _rot: np.zeros(_JOINTS), current_eef=_MEASURED_EEF):
    return mapping.wire_command_to_arm_action(command, current, ik=ik, current_eef=current_eef)


def test_grip_qpos_normalization():
    closed = mapping.GRIPPER_QPOS_CLOSED
    assert mapping.normalize_grip_qpos(0.0) == 0.0
    assert abs(mapping.normalize_grip_qpos(closed / 2) - 0.5) < 1e-6
    assert abs(mapping.normalize_grip_qpos(closed) - 1.0) < 1e-6
    assert mapping.normalize_grip_qpos(closed * 2) == 1.0  # saturates, never exceeds 1
    # A two-finger qpos reads the first finger.
    assert abs(mapping.normalize_grip_qpos(np.array([closed / 2, closed / 2])) - 0.5) < 1e-6


def test_grip_command_to_actuator():
    assert mapping.grip_command_to_actuator(0.0) == 0.0
    assert mapping.grip_command_to_actuator(1.0) == mapping.ROBOTIQ_CLOSED == 255.0
    assert mapping.grip_command_to_actuator(0.5) == 127.5  # continuous — the codec owns binarization
    assert mapping.grip_command_to_actuator(2.0) == 255.0  # clipped


def test_wire_command_joint_pos_passthrough():
    current = np.arange(_JOINTS, dtype=np.float32)
    q = np.full(_JOINTS, 0.3, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: q}, current)
    assert out.dtype == np.float32 and out.shape == (_JOINTS,)
    assert np.array_equal(out, q)  # absolute target, independent of the measured joints


def test_wire_command_joint_vel_integrates_onto_measured():
    current = np.arange(_JOINTS, dtype=np.float32)
    dq = np.full(_JOINTS, 0.1, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.JOINT_VEL, protocol.COMMAND_JOINT_VEL: dq}, current)
    assert np.allclose(out, current + dq)  # positronic applies JointDelta as q + dq


def test_wire_command_hold_recommands_measured():
    current = np.linspace(-1.0, 1.0, _JOINTS, dtype=np.float32)
    out = _action({protocol.COMMAND_TYPE: protocol.HOLD}, current)
    assert np.array_equal(out, current)


def test_wire_command_joint_count_mismatch_raises():
    current = np.zeros(_JOINTS, dtype=np.float32)
    with pytest.raises(ValueError):
        _action(
            {protocol.COMMAND_TYPE: protocol.JOINT_VEL, protocol.COMMAND_JOINT_VEL: np.zeros(6, dtype=np.float32)},
            current,
        )


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
    # The delta is relative to the *measured* pose, so the solver must see the composed absolute target.
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
    with pytest.raises(ValueError, match='cartesian'):  # the message lists the contract the tag is not part of
        _action({protocol.COMMAND_TYPE: 'wrench'}, np.zeros(_JOINTS))


@pytest.mark.parametrize('command_type', protocol.CANONICAL_COMMAND_TYPES)
def test_every_canonical_command_type_converts_to_joint_targets(command_type):
    """The contract is total, so every canonical type converts to the joint targets MolmoSpaces natively steps.
    This is the model-free half of that property — the routing, through a stub solver; ``validate.py`` drives
    the same types through the real IK against a live scene."""
    pose = np.concatenate([np.zeros(3), np.eye(3).reshape(-1)])
    payload = {
        protocol.JOINT_POS: {protocol.COMMAND_JOINT_POS: np.zeros(_JOINTS)},
        protocol.JOINT_VEL: {protocol.COMMAND_JOINT_VEL: np.zeros(_JOINTS)},
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


def test_episode_seed_prefers_the_override_then_the_spec_then_the_index():
    spec = types.SimpleNamespace(seed=7)
    assert mapping.resolve_episode_seed(spec, 3, 99) == 99
    assert mapping.resolve_episode_seed(spec, 3) == 7
    assert mapping.resolve_episode_seed(types.SimpleNamespace(seed=None), 3) == 3
    assert mapping.resolve_episode_seed(types.SimpleNamespace(), 5) == 5


def _lay_out(assets: Path, *relative: str) -> None:
    for path in relative:
        (assets / mapping.ASSETS_BENCHMARKS_DIR / path).mkdir(parents=True)
        (assets / mapping.ASSETS_BENCHMARKS_DIR / path / mapping.MOLMO_BENCHMARK_MANIFEST).write_text('[]')


def test_a_benchmark_path_is_its_four_segments_under_the_benchmarks_root(tmp_path: Path):
    bench = mapping.BenchmarkPath.parse('v1/procthor-10k/Pick/pick_20251231')
    assert bench == mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')
    assert bench.relative == Path('v1/procthor-10k/Pick/pick_20251231')
    assert bench.under(tmp_path) == tmp_path / 'benchmarks/v1/procthor-10k/Pick/pick_20251231'
    with pytest.raises(ValueError, match='suite/scene_dataset/task_config/benchmark'):
        mapping.BenchmarkPath.parse('procthor-10k/Pick/pick_20251231')


def test_discovery_finds_every_manifest_under_the_benchmarks_root(tmp_path: Path):
    _lay_out(tmp_path, 'v2/objaverse/PickHard/hard_20260206', 'v1/procthor-10k/Pick/pick_20251231')
    assert mapping.discover_benchmarks(tmp_path) == [
        mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231'),
        mapping.BenchmarkPath('v2', 'objaverse', 'PickHard', 'hard_20260206'),
    ]


def test_selection_pins_any_dimension_by_a_name_or_a_list_and_leaves_the_rest_open():
    pick_v1 = mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')
    pick_v2 = mapping.BenchmarkPath('v2', 'procthor-10k', 'Pick', 'pick_20251231')
    hard_v2 = mapping.BenchmarkPath('v2', 'objaverse', 'PickHard', 'hard_20260206')
    found = [pick_v1, pick_v2, hard_v2]
    assert mapping.select_benchmarks(found, {}) == found
    assert mapping.select_benchmarks(found, {'suite': 'v2'}) == [pick_v2, hard_v2]
    assert mapping.select_benchmarks(found, {'suite': 'v2', 'task_config': 'Pick'}) == [pick_v2]
    assert mapping.select_benchmarks(found, {'scene_dataset': ['objaverse', 'nowhere']}) == [hard_v2]
    assert mapping.select_benchmarks(found, {'episodes': [0, 1]}) == found


def test_a_selection_matching_no_benchmark_lists_what_is_there():
    found = [mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')]
    with pytest.raises(ValueError, match=r"\{'suite': 'v3'\}.*v1/procthor-10k/Pick/pick_20251231"):
        mapping.select_benchmarks(found, {'suite': 'v3'})
    with pytest.raises(ValueError, match='available under benchmarks/: none'):
        mapping.select_benchmarks([], {})
