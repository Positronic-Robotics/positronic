"""Which tasks and camera sets a RoboLab env server holds once it has task libraries: a library adds them by name."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from positronic.simulator.robolab.task_catalog import METADATA, TaskCatalog, camera_sets

CATEGORIES = {'counting': 'relational', 'color': 'visual'}


def _entry(name: str, attributes: str, episode_s: str = '60', filename: str = 'x.py') -> dict[str, Any]:
    return {
        'task_name': name,
        'episode_s': episode_s,
        'attributes': attributes,
        'filename': filename,
        'instruction': f'do {name}',
        'instruction_variants': {},
    }


def _write(root: Path, entries: list[dict[str, str]]) -> Path:
    (root / METADATA).parent.mkdir(parents=True)
    (root / METADATA).write_text(json.dumps(entries))
    return root


@pytest.fixture
def benchmark(tmp_path) -> Path:
    return _write(tmp_path / 'robolab', [_entry('BananaInBowlTask', 'color'), _entry('CountTask', 'counting')])


@pytest.fixture
def library(tmp_path) -> Path:
    lib = _write(tmp_path / 'lib', [_entry('LabTask', 'counting', '300', 'lab_task.py')])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / 'lab_task.py').write_text(_task_class('LabTask'))
    return lib


def _task_class(name: str) -> str:
    return f'class {name}:\n    pass\n'


def test_a_library_task_runs_by_its_name_with_its_own_budget(benchmark, library):
    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)

    (task,) = catalog.select('LabTask')

    assert task.episode_length_s == 300.0
    assert task.instruction == 'do LabTask'
    assert catalog.register_target('LabTask') == str(library / 'tasks' / 'lab_task.py')
    assert catalog.register_target('CountTask') == 'CountTask'


def test_no_category_and_no_whole_benchmark_sweep_picks_up_a_library_task(benchmark, library):
    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)

    assert [t.name for t in catalog.select(None)] == ['BananaInBowlTask', 'CountTask']
    assert [t.name for t in catalog.select('relational')] == ['CountTask']


def _library(root: Path, name: str, filename: str) -> Path:
    lib = _write(root, [_entry(name, 'counting', filename=filename)])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / filename).write_text(_task_class(name))
    return lib


def test_a_library_task_named_as_a_benchmark_task_fails_at_load_and_names_both(benchmark, tmp_path):
    lib = _library(tmp_path / 'clash', 'CountTask', 'count_task.py')

    with pytest.raises(ValueError) as refused:
        TaskCatalog.load(benchmark, [lib], CATEGORIES)

    message = str(refused.value)
    assert str(lib / 'tasks' / 'count_task.py') in message
    assert message.count("'CountTask'") == 1
    assert 'RoboLab benchmark' in message


def test_two_libraries_may_not_define_one_task_and_the_error_names_both_files(benchmark, tmp_path):
    first = _library(tmp_path / 'first', 'LabTask', 'lab_task.py')
    second = _library(tmp_path / 'second', 'LabTask', 'other_lab_task.py')

    with pytest.raises(ValueError) as refused:
        TaskCatalog.load(benchmark, [first, second], CATEGORIES)

    message = str(refused.value)
    assert str(first / 'tasks' / 'lab_task.py') in message
    assert str(second / 'tasks' / 'other_lab_task.py') in message


def test_a_library_entry_without_its_task_file_fails_at_load(benchmark, tmp_path):
    lib = _write(tmp_path / 'broken', [_entry('LostTask', 'counting', filename='lost.py')])

    with pytest.raises(FileNotFoundError, match='lost.py'):
        TaskCatalog.load(benchmark, [lib], CATEGORIES)


def test_an_unknown_name_names_itself_in_the_error(benchmark, library):
    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)

    with pytest.raises(ValueError, match='NoSuchTask'):
        catalog.select(['LabTask', 'NoSuchTask'])


def test_a_task_file_whose_class_is_not_the_metadata_name_fails_at_load(benchmark, tmp_path):
    lib = _write(tmp_path / 'renamed', [_entry('LabTask', 'counting', filename='lab_task.py')])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / 'lab_task.py').write_text(_task_class('BananaInBowlTask'))

    with pytest.raises(ValueError, match="no module-level 'LabTask'"):
        TaskCatalog.load(benchmark, [lib], CATEGORIES)


SELECTABLE = {'wrist_left', 'exterior_pair_wrist'}


def _cameras(lib: Path, sets: dict[str, list[str]]) -> Path:
    lib.mkdir(parents=True, exist_ok=True)
    (lib / 'cameras.py').write_text(f'def camera_sets():\n    return {sets!r}\n')
    return lib


def test_a_library_camera_set_joins_the_presets(tmp_path):
    lib = _cameras(tmp_path / 'lib', {'exterior_pair_wrist': ['ext', 'ext_2', 'wrist']})
    plain = tmp_path / 'plain'
    plain.mkdir()

    sets = camera_sets([lib, plain], {'wrist_left': ['left', 'wrist']}, SELECTABLE)

    assert sets == {'wrist_left': ['left', 'wrist'], 'exterior_pair_wrist': ['ext', 'ext_2', 'wrist']}


def test_a_catalog_and_its_camera_sets_hold_the_benchmark_and_a_library(benchmark, library):
    _cameras(library, {'exterior_pair_wrist': ['ext', 'ext_2', 'wrist']})

    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)
    sets = camera_sets([library], {'wrist_left': ['left', 'wrist']}, SELECTABLE)

    assert [t.name for t in catalog.select(['CountTask', 'LabTask'])] == ['CountTask', 'LabTask']
    assert sorted(sets) == ['exterior_pair_wrist', 'wrist_left']


def test_a_library_camera_set_may_not_replace_a_preset(tmp_path):
    lib = _cameras(tmp_path / 'lib', {'wrist_left': ['ext', 'wrist']})

    with pytest.raises(ValueError, match="'wrist_left'") as refused:
        camera_sets([lib], {'wrist_left': ['left', 'wrist']}, SELECTABLE)

    assert str(lib / 'cameras.py') in str(refused.value)


def test_two_libraries_may_not_return_one_camera_set_and_the_error_names_both_files(tmp_path):
    first = _cameras(tmp_path / 'first', {'exterior_pair_wrist': ['a']})
    second = _cameras(tmp_path / 'second', {'exterior_pair_wrist': ['b']})

    with pytest.raises(ValueError) as refused:
        camera_sets([first, second], {}, SELECTABLE)

    message = str(refused.value)
    assert str(first / 'cameras.py') in message
    assert str(second / 'cameras.py') in message


def test_a_library_camera_set_that_no_run_can_select_fails_at_load(tmp_path):
    lib = _cameras(tmp_path / 'lib', {'top_down': ['top']})

    with pytest.raises(ValueError, match="'top_down'") as refused:
        camera_sets([lib], {}, SELECTABLE)

    assert str(lib / 'cameras.py') in str(refused.value)


def test_a_library_task_named_as_a_category_fails_at_load(benchmark, tmp_path):
    lib = _library(tmp_path / 'category', 'relational', 'relational.py')

    with pytest.raises(ValueError, match="'relational'"):
        TaskCatalog.load(benchmark, [lib], CATEGORIES)


def test_a_library_task_named_as_an_attribute_outside_the_categories_runs_by_its_name(benchmark, tmp_path):
    lib = _library(tmp_path / 'attribute', 'counting', 'counting.py')

    catalog = TaskCatalog.load(benchmark, [lib], CATEGORIES)

    assert [t.name for t in catalog.select('counting')] == ['counting']


@pytest.mark.parametrize(
    'source', ['LabTask = make_task("lab")\n', 'LabTask: type = make_task("lab")\n'], ids=['assign', 'annotated']
)
def test_a_task_file_that_binds_the_task_name_at_module_level_loads(benchmark, tmp_path, source):
    lib = _write(tmp_path / 'bound', [_entry('LabTask', 'counting', filename='lab_task.py')])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / 'lab_task.py').write_text(source)

    catalog = TaskCatalog.load(benchmark, [lib], CATEGORIES)

    assert catalog.register_target('LabTask') == str(lib / 'tasks' / 'lab_task.py')


@pytest.mark.parametrize(
    'source',
    ['def build():\n    LabTask = make_task("lab")\n', 'registry.LabTask = make_task("lab")\n'],
    ids=['in-a-function', 'attribute'],
)
def test_a_task_file_that_binds_the_task_name_only_below_module_level_fails_at_load(benchmark, tmp_path, source):
    lib = _write(tmp_path / 'unbound', [_entry('LabTask', 'counting', filename='lab_task.py')])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / 'lab_task.py').write_text(source)

    with pytest.raises(ValueError, match="'LabTask'"):
        TaskCatalog.load(benchmark, [lib], CATEGORIES)


def test_a_camera_module_with_a_dataclass_of_postponed_annotations_loads(tmp_path):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'cameras.py').write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass\n\n\n'
        '@dataclass\n'
        'class Camera:\n'
        '    name: str\n\n\n'
        'def camera_sets():\n'
        "    return {'exterior_pair_wrist': [Camera('ext')]}\n"
    )

    sets = camera_sets([lib], {}, SELECTABLE)

    assert [camera.name for camera in sets['exterior_pair_wrist']] == ['ext']


def test_a_camera_module_imports_a_file_beside_it_by_a_relative_import(tmp_path):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'camera_configs.py').write_text("EXTERIORS = ['ext', 'ext_2']\n")
    (lib / 'cameras.py').write_text(
        'from .camera_configs import EXTERIORS\n\n\n'
        'def camera_sets():\n'
        "    return {'exterior_pair_wrist': [*EXTERIORS, 'wrist']}\n"
    )
    path_before = list(sys.path)

    sets = camera_sets([lib], {}, SELECTABLE)

    assert sets == {'exterior_pair_wrist': ['ext', 'ext_2', 'wrist']}
    assert sys.path == path_before


def test_a_camera_module_that_fails_to_load_leaves_no_module_behind(tmp_path):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'cameras.py').write_text("raise RuntimeError('broken camera module')\n")

    with pytest.raises(RuntimeError, match='broken camera module'):
        camera_sets([lib], {}, SELECTABLE)

    assert [m for m in list(sys.modules.values()) if getattr(m, '__file__', None) == str(lib / 'cameras.py')] == []
