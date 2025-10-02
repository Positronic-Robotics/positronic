from pathlib import Path

import numpy as np
import pytest

from positronic.dataset import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter


def test_local_dataset_writer_creates_structure_and_persists(tmp_path):
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as w:
        # Create three episodes with minimal content
        for i in range(3):
            with w.new_episode() as ew:
                ew.set_static('id', i)
                ew.append('a', i, 1000 + i)

    # Structure exists (12-digit zero-padded ids)
    assert (root / '000000000000' / '000000000000').exists()
    assert (root / '000000000000' / '000000000001').exists()
    assert (root / '000000000000' / '000000000002').exists()
    assert (root / 'signals_meta.json').exists()

    ds = LocalDataset(root)
    assert len(ds) == 3
    ep0 = ds[0]
    assert isinstance(ep0, Episode)
    assert ep0['id'] == 0
    assert ep0['a'][0] == (0, 1000)
    ds_meta = ds.signals_meta
    assert 'a' in ds_meta

    # Restart writer and keep appending
    with LocalDatasetWriter(root) as w2:
        with w2.new_episode() as ew:
            ew.set_static('id', 3)

    ds2 = LocalDataset(root)
    assert len(ds2) == 4
    assert ds2[3]['id'] == 3


def test_local_dataset_writer_appends_existing(tmp_path):
    root = tmp_path / 'append'

    writer = LocalDatasetWriter(root)
    with writer.new_episode() as episode:
        episode.append('test_signal', np.array([1.0], dtype=np.float32), ts_ns=1)

    writer = LocalDatasetWriter(root)
    with writer.new_episode() as episode:
        episode.append('test_signal', np.array([2.0], dtype=np.float32), ts_ns=2)

    ds = LocalDataset(root)
    assert len(ds) == 2

    first_signal = ds[0]['test_signal']
    second_signal = ds[1]['test_signal']

    np.testing.assert_allclose(first_signal[0][0], np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(second_signal[0][0], np.array([2.0], dtype=np.float32))

    block_dir = root / '000000000000'
    assert (block_dir / '000000000000').exists()
    assert (block_dir / '000000000001').exists()


def test_local_dataset_handles_block_rollover(tmp_path):
    root = tmp_path / 'roll'
    with LocalDatasetWriter(root) as w:
        # Create 1001 empty episodes (static-only) to cross a block boundary
        for i in range(1001):
            with w.new_episode() as ew:
                ew.set_static('id', i)

    # Check directories for episode 0 and 1000
    assert (root / '000000000000' / '000000000000').exists()
    assert (root / '000000001000' / '000000001000').exists()

    ds = LocalDataset(root)
    assert len(ds) == 1001
    assert ds[0]['id'] == 0
    assert ds[1000]['id'] == 1000


# --- Indexing behavior tests ---


def build_simple_dataset(root: Path, n: int = 5) -> LocalDataset:
    with LocalDatasetWriter(root) as w:
        for i in range(n):
            with w.new_episode() as ew:
                ew.set_static('id', i)
    return LocalDataset(root)


def episode_ids(episodes):
    return [ep['id'] for ep in episodes]


def test_slice_indexing_returns_episode_list(tmp_path):
    ds = build_simple_dataset(tmp_path / 'ds', n=5)

    sub = ds[1:4]
    assert isinstance(sub, list)
    assert len(sub) == 3
    assert all(isinstance(ep, Episode) for ep in sub)
    assert episode_ids(sub) == [1, 2, 3]

    sub2 = ds[0:5:2]
    assert episode_ids(sub2) == [0, 2, 4]

    # Negative step slice
    sub3 = ds[4:1:-1]
    assert episode_ids(sub3) == [4, 3, 2]


def test_array_indexing_returns_episode_list(tmp_path):
    ds = build_simple_dataset(tmp_path / 'ds2', n=5)

    idx_list = [0, 3, 1]
    out = ds[idx_list]
    assert isinstance(out, list)
    assert episode_ids(out) == [0, 3, 1]

    idx_np = np.array([4, 0], dtype=int)
    out2 = ds[idx_np]
    assert episode_ids(out2) == [4, 0]

    # Negative indices
    out3 = ds[[-1, -5]]
    assert episode_ids(out3) == [4, 0]


def test_array_indexing_errors(tmp_path):
    ds = build_simple_dataset(tmp_path / 'ds3', n=4)

    # Boolean mask not supported
    with np.testing.assert_raises_regex(TypeError, 'Boolean indexing is not supported'):
        _ = ds[[True, False, True, False]]

    with np.testing.assert_raises_regex(TypeError, 'Boolean indexing is not supported'):
        _ = ds[np.array([True, False, True, False])]

    # Out of range
    with np.testing.assert_raises(IndexError):
        _ = ds[[10]]


def test_signals_meta_fallback_without_file(tmp_path):
    root = tmp_path / 'meta_missing'
    with LocalDatasetWriter(root) as w:
        with w.new_episode() as ew:
            ew.append('b', 1, 1000)

    (root / 'signals_meta.json').unlink()

    ds = LocalDataset(root)
    meta = ds.signals_meta
    assert 'b' in meta


def test_homedir_resolution(tmp_path):
    # Create a test dataset under home directory
    import tempfile

    home = Path.home()
    with tempfile.TemporaryDirectory(dir=home) as tmpdir:
        actual_root = Path(tmpdir) / 'ds'
        with LocalDatasetWriter(actual_root) as w:
            with w.new_episode() as ew:
                ew.set_static('id', 42)

        # Test LocalDataset with ~ path
        relative_to_home = actual_root.relative_to(home)
        tilde_path = Path('~') / relative_to_home

        ds = LocalDataset(tilde_path)
        assert len(ds) == 1
        assert ds[0]['id'] == 42

        # Test LocalDatasetWriter with ~ path
        with LocalDatasetWriter(tilde_path) as w:
            with w.new_episode() as ew:
                ew.set_static('id', 43)

        ds2 = LocalDataset(actual_root)
        assert len(ds2) == 2
        assert ds2[1]['id'] == 43


def test_local_dataset_requires_existing_root(tmp_path):
    missing_root = tmp_path / 'missing_dataset'

    with pytest.raises(FileNotFoundError) as excinfo:
        LocalDataset(missing_root)

    assert str(missing_root) in str(excinfo.value)
