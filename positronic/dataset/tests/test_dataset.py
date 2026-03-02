from pathlib import Path

import numpy as np

from positronic.dataset.dataset import ConcatDataset, FilterDataset
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter


def episode_ids(episodes):
    return [ep['id'] for ep in episodes]


def build_dataset_with_signal(root: Path, values: list[int]) -> LocalDataset:
    with LocalDatasetWriter(root) as w:
        for i, value in enumerate(values):
            with w.new_episode() as ew:
                ew.set_static('id', value)
                ew.append('signal', np.array([value], dtype=np.float32), ts_ns=10_000 + i)
    return LocalDataset(root)


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
