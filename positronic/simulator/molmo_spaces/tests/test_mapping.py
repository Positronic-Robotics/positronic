"""Benchmark selection and seed handling, checked without the MolmoSpaces runtime."""

import types
from pathlib import Path

import numpy as np
import pytest
from platform_client.evals import MOLMO_CONFIG, PUBLIC_EVALS, public_eval
from scipy.spatial.transform import Rotation

from positronic.simulator.molmo_spaces import mapping


def test_grip_qpos_normalization():
    closed = mapping.GRIPPER_QPOS_CLOSED
    assert mapping.normalize_grip_qpos(0.0) == 0.0
    assert abs(mapping.normalize_grip_qpos(closed / 2) - 0.5) < 1e-6
    assert abs(mapping.normalize_grip_qpos(closed) - 1.0) < 1e-6
    assert mapping.normalize_grip_qpos(closed * 2) == 1.0
    assert abs(mapping.normalize_grip_qpos(np.array([closed / 2, closed / 2])) - 0.5) < 1e-6


def test_grip_command_to_actuator():
    assert mapping.grip_command_to_actuator(0.0) == 0.0
    assert mapping.grip_command_to_actuator(1.0) == mapping.ROBOTIQ_CLOSED == 255.0
    assert mapping.grip_command_to_actuator(0.5) == 127.5
    assert mapping.grip_command_to_actuator(2.0) == 255.0


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


def test_discovery_finds_a_benchmark_behind_a_symlinked_suite_directory(tmp_path: Path):
    # MolmoSpaces' asset manager links each suite to one version of its cache: benchmarks/<suite> -> <cache>/<version>.
    version = tmp_path / 'cache' / 'v1' / '20260408'
    (version / 'procthor-10k' / 'Pick' / 'pick_20251231').mkdir(parents=True)
    (version / 'procthor-10k' / 'Pick' / 'pick_20251231' / mapping.MOLMO_BENCHMARK_MANIFEST).write_text('[]')
    (tmp_path / 'assets' / mapping.ASSETS_BENCHMARKS_DIR).mkdir(parents=True)
    (tmp_path / 'assets' / mapping.ASSETS_BENCHMARKS_DIR / 'v1').symlink_to(version, target_is_directory=True)
    assert mapping.discover_benchmarks(tmp_path / 'assets') == [
        mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')
    ]


def test_discovery_does_not_walk_a_symlink_back_into_its_own_ancestor(tmp_path: Path):
    _lay_out(tmp_path, 'v1/procthor-10k/Pick/pick_20251231')
    suite = tmp_path / mapping.ASSETS_BENCHMARKS_DIR / 'v1'
    (suite / 'procthor-10k' / 'loop').symlink_to(suite, target_is_directory=True)
    assert mapping.discover_benchmarks(tmp_path) == [
        mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')
    ]


def test_discovery_raises_on_a_directory_it_cannot_read(tmp_path: Path):
    _lay_out(tmp_path, 'v1/procthor-10k/Pick/pick_20251231')
    unreadable = tmp_path / mapping.ASSETS_BENCHMARKS_DIR / 'v1' / 'procthor-10k'
    unreadable.chmod(0)
    try:
        with pytest.raises(PermissionError):
            mapping.discover_benchmarks(tmp_path)
    finally:
        unreadable.chmod(0o755)


def test_discovery_raises_when_the_asset_directory_holds_no_benchmarks_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        mapping.discover_benchmarks(tmp_path)


def test_selection_pins_any_dimension_by_a_name_or_a_list_and_leaves_the_rest_open():
    pick_v1 = mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')
    pick_v2 = mapping.BenchmarkPath('v2', 'procthor-10k', 'Pick', 'pick_20251231')
    hard_v2 = mapping.BenchmarkPath('v2', 'objaverse', 'PickHard', 'hard_20260206')
    found = [pick_v1, pick_v2, hard_v2]
    assert mapping.select_benchmarks(found, {}) == found
    assert mapping.select_benchmarks(found, {'suite': 'v2'}) == [pick_v2, hard_v2]
    assert mapping.select_benchmarks(found, {'suite': 'v2', 'task_config': 'Pick'}) == [pick_v2]
    assert mapping.select_benchmarks(found, {'scene_dataset': ['objaverse', 'nowhere']}) == [hard_v2]
    assert mapping.select_benchmarks(found, {mapping.SELECT_EPISODES: [0, 1]}) == found


def test_a_selection_matching_no_benchmark_lists_what_is_there():
    found = [mapping.BenchmarkPath('v1', 'procthor-10k', 'Pick', 'pick_20251231')]
    with pytest.raises(ValueError, match=r"\{'suite': 'v3'\}.*v1/procthor-10k/Pick/pick_20251231"):
        mapping.select_benchmarks(found, {'suite': 'v3'})
    with pytest.raises(ValueError, match='available under benchmarks/: none'):
        mapping.select_benchmarks([], {})


# The nominal left shoulder pose of MolmoSpaces' Franka DROID camera systems, in the robot base frame (wxyz), and the
# randomization FrankaOmniPurposeCameraSystem gives that camera.
_LEFT_OFFSET = (0.1, 0.57, 0.66)
_LEFT_QUAT = (-0.3633, -0.1241, 0.4263, 0.8191)
_POS_NOISE, _ORIENTATION_NOISE_DEG = (-0.05, 0.05), 8.0
_HARD = mapping.BenchmarkPath('v2', 'procthor-10k', 'FrankaPickHardBench', 'hard_20260212')
_REFLECT_Y = np.array([1.0, -1.0, 1.0])


def _axis_on_the_base_plane(offset, quat) -> tuple[np.ndarray, np.ndarray]:
    """Where the camera's optical axis meets the robot base's z = 0 plane, and the camera's up direction."""
    rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    forward, up = -rotation[:, 2], rotation[:, 1]
    reach = -offset[2] / forward[2]
    assert reach > 0, 'the camera looks away from the plane'
    return np.asarray(offset) + reach * forward, up


def _right_pose(bench: mapping.BenchmarkPath, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    offset, quat = mapping.mirror_camera_pose(_LEFT_OFFSET, _LEFT_QUAT)
    rng = mapping.episode_rng(bench, episode_index)
    return mapping.jitter_camera_pose(offset, quat, _POS_NOISE, _ORIENTATION_NOISE_DEG, rng)


def test_the_mirrored_shoulder_camera_looks_at_the_mirrored_point_on_the_table():
    offset, quat = mapping.mirror_camera_pose(_LEFT_OFFSET, _LEFT_QUAT)
    np.testing.assert_allclose(offset, [0.1, -0.57, 0.66])
    np.testing.assert_allclose(quat, [0.8191, 0.4263, -0.1241, -0.3633], atol=1e-4)
    left_target, left_up = _axis_on_the_base_plane(np.array(_LEFT_OFFSET), _LEFT_QUAT)
    right_target, right_up = _axis_on_the_base_plane(offset, quat)
    np.testing.assert_allclose(right_target, left_target * _REFLECT_Y, atol=1e-9)
    np.testing.assert_allclose(right_up, left_up * _REFLECT_Y, atol=1e-9)
    assert right_target[0] > 0 and right_up[2] > 0


def test_a_fixed_left_camera_leaves_the_right_camera_at_the_mirrored_nominal_pose():
    offset, quat = mapping.mirror_camera_pose(_LEFT_OFFSET, _LEFT_QUAT)
    fixed_offset, fixed_quat = mapping.jitter_camera_pose(offset, quat, None, None, mapping.episode_rng(_HARD, 3))
    np.testing.assert_allclose(fixed_offset, offset)
    np.testing.assert_allclose(fixed_quat, quat)


def test_an_episode_draws_one_right_camera_pose_and_other_episodes_draw_their_own():
    offset, quat = _right_pose(_HARD, 3)
    again_offset, again_quat = _right_pose(_HARD, 3)
    np.testing.assert_array_equal(offset, again_offset)
    np.testing.assert_array_equal(quat, again_quat)
    assert not np.allclose(offset, _right_pose(_HARD, 4)[0])
    assert not np.allclose(offset, _right_pose(_HARD._replace(suite='v1'), 3)[0])


def test_the_right_camera_draw_leaves_the_global_numpy_random_state_alone():
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    _right_pose(_HARD, 3)
    assert np.random.random() == expected


def test_the_right_camera_jitter_takes_the_left_camera_ranges_in_the_robot_base_frame():
    nominal_offset, nominal_quat = mapping.mirror_camera_pose(_LEFT_OFFSET, _LEFT_QUAT)
    nominal = Rotation.from_quat(nominal_quat, scalar_first=True)
    poses = [_right_pose(_HARD, i) for i in range(500)]
    shifts = np.abs([offset - nominal_offset for offset, _ in poses])
    turns = np.abs([
        (Rotation.from_quat(q, scalar_first=True) * nominal.inv()).as_euler('xyz', degrees=True) for _, q in poses
    ])
    assert 0.045 < shifts.max() <= _POS_NOISE[1]
    assert 7.5 < turns.max() <= _ORIENTATION_NOISE_DEG + 1e-9


_BLOCKED = frozenset({3})


def test_an_open_selection_leaves_out_the_blocked_episodes():
    assert 3 not in mapping.select_episodes(_HARD, 6, None, _BLOCKED)


def test_an_open_selection_keeps_every_clear_episode():
    assert mapping.select_episodes(_HARD, 6, None, _BLOCKED) == [0, 1, 2, 4, 5]


def test_a_pinned_selection_runs_its_clear_episodes_in_its_order():
    assert mapping.select_episodes(_HARD, 6, [5, 1], _BLOCKED) == [5, 1]


def test_a_pinned_blocked_episode_is_refused():
    with pytest.raises(ValueError, match=r'episodes \[3\] of .* hide the start grasp site'):
        mapping.select_episodes(_HARD, 6, [1, 3], _BLOCKED)


def test_a_pinned_episode_out_of_range_is_refused():
    with pytest.raises(ValueError, match=r'episodes \[6\] out of range for the 6 episodes'):
        mapping.select_episodes(_HARD, 6, [0, 6], frozenset())


def test_no_public_eval_pins_an_episode_the_committed_list_blocks():
    blocked = mapping.load_right_view_blocked()
    for name in PUBLIC_EVALS:
        definition = public_eval(name)
        if definition.config != MOLMO_CONFIG:
            continue
        bench = mapping.BenchmarkPath(*(str(definition.args[d]) for d in mapping.BenchmarkPath._fields))
        episodes = definition.args[mapping.SELECT_EPISODES]
        assert isinstance(episodes, list)
        assert not set(episodes) & blocked.get(bench, frozenset()), name
