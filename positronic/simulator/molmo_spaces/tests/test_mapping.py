"""Benchmark selection and seed handling, checked without the MolmoSpaces runtime."""

import types
from pathlib import Path

import numpy as np
import pytest

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
