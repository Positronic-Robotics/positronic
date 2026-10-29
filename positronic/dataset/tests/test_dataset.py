import pickle
from pathlib import Path

import numpy as np
import pytest

from positronic.cfg import dataset_writer, video_encoder
from positronic.dataset import Time
from positronic.dataset.dataset import ConcatDataset, FilterDataset
from positronic.dataset.edits import EditedDataset
from positronic.dataset.episode import META_WRITER_VIDEO_ENCODER
from positronic.dataset.gst_video import GstH264Encoder
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.vector import RECORDED_TIME
from positronic.dataset.video import DEFAULT_VIDEO_ENCODER, LibavEncoder


def test_writer_rejects_an_unavailable_encoder_before_creating_the_dataset(tmp_path):
    class AbsentEncoder(LibavEncoder):
        def ensure_available(self) -> None:
            raise RuntimeError('no such encoder here')

    root = tmp_path / 'recording'
    with pytest.raises(RuntimeError, match='no such encoder here'):
        with LocalDatasetWriter(root, video_encoder=AbsentEncoder()):
            pass
    assert not root.exists()


def episode_ids(episodes):
    return [ep['id'] for ep in episodes]


def build_dataset_with_signal(root: Path, values: list[int]) -> LocalDataset:
    with LocalDatasetWriter(root) as w:
        for i, value in enumerate(values):
            with w.new_episode() as ew:
                ew.set_static('id', value)
                ew.append('signal', np.array([value], dtype=np.float32), Time(**{RECORDED_TIME: 10_000 + i}))
    return LocalDataset(root)


# --- EditedDataset tests ---


def test_edited_dataset_applies_edits_by_uid(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0, 1])
    edited = EditedDataset(ds).set_static(ds[0].meta['uid'], {'id': 100, 'verdict': 'success'})

    assert len(edited) == 2
    assert edited[0]['id'] == 100
    assert edited[0]['verdict'] == 'success'
    assert edited[1]['id'] == 1
    assert 'verdict' not in edited[1]


# --- ConcatDataset tests ---


def test_concat_dataset_length_and_indexing(tmp_path):
    ds1 = build_dataset_with_signal(tmp_path / 'concat_ds1', [0, 1])
    ds2 = build_dataset_with_signal(tmp_path / 'concat_ds2', [2, 3, 4])

    concatenated = ConcatDataset(ds1, ds2)

    assert len(concatenated) == 5
    ids = [concatenated[i]['id'] for i in range(len(concatenated))]
    assert ids == [0, 1, 2, 3, 4]

    second_half = concatenated[3:]
    assert episode_ids(second_half) == [3, 4]


def test_dataset_add_operator_returns_concat(tmp_path):
    ds1 = build_dataset_with_signal(tmp_path / 'add_ds1', [0])
    ds2 = build_dataset_with_signal(tmp_path / 'add_ds2', [1, 2])

    combined = ds1 + ds2
    assert isinstance(combined, ConcatDataset)
    assert episode_ids(combined[:]) == [0, 1, 2]


# --- FilterDataset tests ---


def test_filter_dataset_subset(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', [0, 1, 2, 3, 4])
    filtered = FilterDataset(ds, lambda ep: ep['id'] % 2 == 0)
    assert len(filtered) == 3
    assert episode_ids(filtered[:]) == [0, 2, 4]


def test_filter_dataset_empty(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', [0, 1, 2])
    filtered = FilterDataset(ds, lambda ep: False)
    assert len(filtered) == 0


def test_filter_dataset_pass_through(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', [0, 1, 2])
    filtered = FilterDataset(ds, lambda ep: True)
    assert len(filtered) == len(ds)
    assert episode_ids(filtered[:]) == [0, 1, 2]


def test_filter_dataset_lazy(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', [0, 1, 2])
    call_count = 0

    def counting_predicate(ep):
        nonlocal call_count
        call_count += 1
        return True

    filtered = FilterDataset(ds, counting_predicate)
    assert call_count == 0

    _ = len(filtered)
    assert call_count == 3


@pytest.mark.parametrize(
    ('config', 'encoder'),
    [(dataset_writer.local, DEFAULT_VIDEO_ENCODER), (dataset_writer.jetson, video_encoder.jetson_h264.instantiate())],
)
def test_writer_presets_survive_process_transfer_and_reach_the_episode(tmp_path, monkeypatch, config, encoder):
    checked = []
    monkeypatch.setattr(GstH264Encoder, 'ensure_available', lambda self: checked.append(self))
    factory = pickle.loads(pickle.dumps(config.instantiate()))
    with factory(tmp_path) as dataset, dataset.new_episode() as episode:
        episode.append('value', 1, Time(tick=0))
    recorded = LocalDataset(tmp_path)[0]
    assert recorded.meta['writer'][META_WRITER_VIDEO_ENCODER] == repr(encoder)
    if isinstance(encoder, GstH264Encoder):
        assert checked == [encoder]
