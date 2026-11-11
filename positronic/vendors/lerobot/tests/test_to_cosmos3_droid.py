import csv
import json
import os

import numpy as np
import pos3
import pyarrow.parquet as pq
import pytest

from positronic import keys
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.time import Time
from positronic.drivers.recording import HARNESS_WORLD
from positronic.vendors.lerobot import cosmos3_droid as droid

lerobot = pytest.importorskip('lerobot')
if not hasattr(lerobot, '__version__') or lerobot.__version__ < '0.4':
    pytest.skip('Requires lerobot >= 0.4', allow_module_level=True)

os.environ['HF_HUB_OFFLINE'] = '1'

from positronic.vendors.lerobot.to_cosmos3_droid import START_TRIM_FILE, TrimRow, convert  # noqa: E402

_FRAMES = 8
_STEP_NS = 1_000_000_000 // droid.FPS


def _write_episode(writer, drops: list[float]) -> None:
    for i, drop in enumerate(drops):
        time = Time(**{HARNESS_WORLD: i * _STEP_NS})
        writer.append(keys.JOINTS, np.full(7, float(i)), time)
        writer.append(keys.GRIP, 0.25 + 0.05 * i, time)
        writer.append(keys.TARGET_GRIP, 0.5, time)
        writer.append(keys.EE_POSE, np.array([0.5, 0.0, 0.6 - drop, 0.0, 1.0, 0.0, 0.0]), time)
        writer.append(keys.EXTERIOR_IMAGE, np.full((36, 64, 3), 20 * i, dtype=np.uint8), time)
        writer.append(keys.WRIST_IMAGE, np.full((36, 64, 3), 10 * i, dtype=np.uint8), time)
    writer.set_static(keys.TASK, 'Put the towels in the grey tote.')


@pytest.fixture
def converted(tmp_path):
    source = tmp_path / 'source'
    with LocalDatasetWriter(source) as dataset:
        with dataset.new_episode() as writer:
            _write_episode(writer, [0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.2, 0.2])
        with dataset.new_episode() as writer:
            _write_episode(writer, [0.0, 0.05, 0.1, 0.149, 0.1, 0.0, 0.0, 0.0])
    output = tmp_path / 'out'
    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        convert(dataset=LocalDataset(source), output_dir=str(output), heldout=None, num_workers=0)
    return output / 'train' / droid.DATASET_DIR


def test_a_kept_episode_starts_at_the_first_frame_15_cm_down(converted):
    data = pq.read_table(next((converted / droid.SUCCESS_DIR / 'data').rglob('*.parquet')))
    joints = np.asarray(data[droid.JOINT_POSITIONS].to_pylist())
    np.testing.assert_array_equal(joints[:, 0], [4, 5])
    np.testing.assert_array_equal(np.asarray(data[droid.ACTION_JOINT_POSITION].to_pylist())[:, 0], [5, 6])
    timestamps = data['timestamp'].to_numpy()
    assert np.array_equal(timestamps.view(np.uint32), droid.frame_timestamps(2).view(np.uint32))


def _trim_rows(converted) -> list[TrimRow]:
    return [TrimRow(**row) for row in csv.DictReader((converted.parent.parent / START_TRIM_FILE).open())]


def test_an_episode_that_never_gets_15_cm_down_is_left_out_and_listed(converted):
    rows = _trim_rows(converted)
    assert [(row.tree, row.cut_frame) for row in rows] == [('train', '4'), ('', '')]
    episodes = pq.read_table(next((converted / droid.SUCCESS_DIR / 'meta' / 'episodes').rglob('*.parquet')))
    assert episodes[droid.EPISODE_ID].to_pylist() == [rows[0].uid]


def test_the_keep_ranges_cover_each_kept_episode_by_its_id(converted):
    ranges = json.loads((converted / droid.KEEP_RANGES_FILE).read_text())
    uid = _trim_rows(converted)[0].uid
    assert ranges == {droid.keep_ranges_key(uid): [[0, 2]]}


def test_one_exterior_camera_fills_both_exterior_views(converted):
    info = json.loads((converted / droid.SUCCESS_DIR / 'meta' / 'info.json').read_text())
    for view in (droid.EXTERIOR_IMAGE_1, droid.EXTERIOR_IMAGE_2, droid.WRIST_IMAGE):
        assert info['features'][view]['shape'] == [360, 640, 3]
    assert info['fps'] == droid.FPS
