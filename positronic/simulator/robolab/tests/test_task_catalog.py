"""Which tasks a RoboLab env server holds once it has task libraries: a library adds tasks by name only."""

import json
from pathlib import Path
from typing import Any

import pytest

from positronic import keys
from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.task_catalog import METADATA, TaskCatalog

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
    (lib / 'tasks' / 'lab_task.py').write_text('')
    return lib


def test_a_library_task_runs_by_its_name_with_its_own_budget(benchmark, library):
    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)

    (task,) = catalog.select('LabTask', set(CATEGORIES.values()))

    assert task.episode_length_s == 300.0
    assert task.instruction == 'do LabTask'
    assert catalog.register_target('LabTask') == str(library / 'tasks' / 'lab_task.py')
    assert catalog.register_target('CountTask') == 'CountTask'


def test_no_category_and_no_whole_benchmark_sweep_picks_up_a_library_task(benchmark, library):
    catalog = TaskCatalog.load(benchmark, [library], CATEGORIES)

    assert [t.name for t in catalog.select(None, set(CATEGORIES.values()))] == ['BananaInBowlTask', 'CountTask']
    assert [t.name for t in catalog.select('relational', set(CATEGORIES.values()))] == ['CountTask']


def _library(root: Path, name: str, filename: str) -> Path:
    lib = _write(root, [_entry(name, 'counting', filename=filename)])
    (lib / 'tasks').mkdir()
    (lib / 'tasks' / filename).write_text('')
    return lib


def test_a_library_task_named_as_a_benchmark_task_fails_at_load_and_names_both(benchmark, tmp_path):
    lib = _library(tmp_path / 'clash', 'CountTask', 'count_task.py')

    with pytest.raises(ValueError) as refused:
        TaskCatalog.load(benchmark, [lib], CATEGORIES)

    message = str(refused.value)
    assert str(lib / 'tasks' / 'count_task.py') in message
    assert "RoboLab benchmark task 'CountTask'" in message


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
        catalog.select(['LabTask', 'NoSuchTask'], set(CATEGORIES.values()))


def test_the_exterior_pair_set_binds_both_exteriors_and_the_wrist():
    wire = robolab_cfg._droid_camera_dict(robolab_keys.camera_set(robolab_keys.DROID, robolab_keys.EXTERIOR_PAIR_WRIST))

    assert wire == {
        keys.EXTERIOR_IMAGE: robolab_keys.EXTERIOR_CAMERA,
        keys.EXTERIOR_IMAGE_2: robolab_keys.EXTERIOR_2_CAMERA,
        keys.WRIST_IMAGE: robolab_keys.WRIST_CAMERA,
    }
