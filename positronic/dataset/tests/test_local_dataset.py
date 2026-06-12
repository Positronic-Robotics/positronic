import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from positronic.dataset import Episode, edits
from positronic.dataset.local_dataset import (
    UNFINISHED_MARKER,
    DiskEpisode,
    LocalDataset,
    LocalDatasetWriter,
    load_all_datasets,
    load_dataset,
)

from .test_dataset import build_dataset_with_signal, episode_ids


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

    ds = LocalDataset(root)
    assert len(ds) == 3
    ep0 = ds[0]
    assert isinstance(ep0, Episode)
    assert ep0['id'] == 0
    assert ep0['a'][0] == (0, 1000)

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


def test_local_dataset_ignores_unfinished_episodes(tmp_path):
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as w:
        with w.new_episode() as ew:
            ew.set_static('id', 0)

    writer = LocalDatasetWriter(root)
    unfinished = writer.new_episode()
    marker = unfinished.path / UNFINISHED_MARKER
    assert marker.exists()

    ds = LocalDataset(root)
    assert len(ds) == 1
    assert ds[0]['id'] == 0


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


def test_slice_indexing_returns_episode_list(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', list(range(5)))

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
    ds = build_dataset_with_signal(tmp_path / 'ds2', list(range(5)))

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
    ds = build_dataset_with_signal(tmp_path / 'ds3', list(range(4)))

    # Boolean mask not supported
    with np.testing.assert_raises_regex(TypeError, 'Boolean indexing is not supported'):
        _ = ds[[True, False, True, False]]

    with np.testing.assert_raises_regex(TypeError, 'Boolean indexing is not supported'):
        _ = ds[np.array([True, False, True, False])]

    # Out of range
    with np.testing.assert_raises(IndexError):
        _ = ds[[10]]


def test_homedir_resolution(tmp_path):
    # Create a test dataset under home directory
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


# --- load_all_datasets tests ---


def test_load_all_datasets_from_root_itself(tmp_path):
    """Test loading when the root directory itself is a dataset."""
    root = tmp_path / 'dataset'

    build_dataset_with_signal(root, [0, 1, 2])

    result = load_all_datasets(root)

    assert len(result) == 3
    assert episode_ids(result[:]) == [0, 1, 2]


def test_load_all_datasets_root_is_dataset_with_subdatasets(tmp_path):
    """Test that if root is a valid dataset, subdirectories are still explored."""
    root = tmp_path / 'datasets'

    # Root itself is a dataset
    build_dataset_with_signal(root, [0, 1])
    # Root also has subdirectories with datasets
    build_dataset_with_signal(root / 'ds1', [2, 3])
    build_dataset_with_signal(root / 'ds2', [4, 5])

    result = load_all_datasets(root)

    # Should load all datasets: root + ds1 + ds2
    assert len(result) == 6
    assert episode_ids(result[:]) == [0, 1, 2, 3, 4, 5]


def test_load_all_datasets_single_dataset(tmp_path):
    """Test loading a directory with a single dataset."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create one dataset
    build_dataset_with_signal(root / 'ds1', [0, 1, 2])

    # Load all datasets
    result = load_all_datasets(root)

    assert len(result) == 3
    assert episode_ids(result[:]) == [0, 1, 2]


def test_load_all_datasets_multiple_datasets(tmp_path):
    """Test loading a directory with multiple datasets."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create three datasets
    build_dataset_with_signal(root / 'ds1', [0, 1])
    build_dataset_with_signal(root / 'ds2', [2, 3, 4])
    build_dataset_with_signal(root / 'ds3', [5])

    # Load all datasets
    result = load_all_datasets(root)

    assert len(result) == 6
    assert episode_ids(result[:]) == [0, 1, 2, 3, 4, 5]


def test_load_all_datasets_skips_empty_datasets(tmp_path):
    """Test that empty datasets are skipped."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create one valid dataset and one empty directory
    build_dataset_with_signal(root / 'ds1', [0, 1])
    (root / 'empty_ds').mkdir()

    # Load all datasets
    result = load_all_datasets(root)
    assert len(result) == 2
    assert episode_ids(result[:]) == [0, 1]


def test_load_all_datasets_skips_non_dataset_directories(tmp_path):
    """Test that non-dataset directories are skipped gracefully."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create a valid dataset
    build_dataset_with_signal(root / 'ds1', [0, 1])

    # Create some non-dataset directories
    (root / 'random_dir').mkdir()
    (root / 'another_dir').mkdir()
    (root / 'random_dir' / 'file.txt').write_text('not a dataset')

    # Load all datasets
    result = load_all_datasets(root)

    assert len(result) == 2
    assert episode_ids(result[:]) == [0, 1]


def test_load_all_datasets_skips_files(tmp_path):
    """Test that regular files in the root are ignored."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create a valid dataset
    build_dataset_with_signal(root / 'ds1', [0, 1, 2])

    # Create some regular files
    (root / 'readme.txt').write_text('readme')
    (root / 'config.json').write_text('{}')

    # Load all datasets
    result = load_all_datasets(root)

    assert len(result) == 3
    assert episode_ids(result[:]) == [0, 1, 2]


def test_load_all_datasets_deep_nesting(tmp_path):
    """Test BFS finds datasets at different depths."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create datasets at different levels
    # Level 1: not a dataset, has subdirs
    (root / 'a').mkdir()
    build_dataset_with_signal(root / 'a' / 'dataset1', [0, 1])

    # Level 1: is a dataset and also has a subdataset
    build_dataset_with_signal(root / 'b', [2, 3])
    build_dataset_with_signal(root / 'b' / 'nested', [99])

    # Level 1: not a dataset, has deeper nesting
    (root / 'c').mkdir()
    (root / 'c' / 'level2').mkdir()
    build_dataset_with_signal(root / 'c' / 'level2' / 'dataset2', [4, 5])

    result = load_all_datasets(root)

    # Should find all datasets: a/dataset1, b, b/nested, c/level2/dataset2
    assert len(result) == 7
    ids = episode_ids(result[:])
    assert sorted(ids) == [0, 1, 2, 3, 4, 5, 99]


def test_load_all_datasets_alphabetical_order(tmp_path):
    """Test that datasets are loaded in alphabetical order."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create datasets with names that would be out of order if not sorted
    build_dataset_with_signal(root / 'ds_c', [6, 7])
    build_dataset_with_signal(root / 'ds_a', [0, 1])
    build_dataset_with_signal(root / 'ds_b', [2, 3, 4, 5])

    # Load all datasets
    result = load_all_datasets(root)

    assert len(result) == 8
    assert episode_ids(result[:]) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_load_all_datasets_nonexistent_directory(tmp_path):
    """Test that FileNotFoundError is raised for non-existent directory."""
    missing_path = tmp_path / 'nonexistent'

    with pytest.raises(FileNotFoundError, match='does not exist'):
        load_all_datasets(missing_path)


def test_load_all_datasets_not_a_directory(tmp_path):
    """Test that ValueError is raised when path is a file, not a directory."""
    file_path = tmp_path / 'file.txt'
    file_path.write_text('not a directory')

    with pytest.raises(ValueError, match='is not a directory'):
        load_all_datasets(file_path)


def test_load_all_datasets_no_valid_datasets(tmp_path):
    """Test that ValueError is raised when no valid datasets are found."""
    root = tmp_path / 'datasets'
    root.mkdir()

    # Create only non-dataset directories
    (root / 'random1').mkdir()
    (root / 'random2').mkdir()
    (root / 'file.txt').write_text('file')

    with pytest.raises(ValueError, match='No valid datasets found'):
        load_all_datasets(root)


def test_load_all_datasets_with_tilde_path(tmp_path):
    """Test that tilde paths are expanded correctly."""
    home = Path.home()
    with tempfile.TemporaryDirectory(dir=home) as tmpdir:
        actual_root = Path(tmpdir) / 'datasets'
        actual_root.mkdir()

        # Create datasets
        build_dataset_with_signal(actual_root / 'ds1', [0, 1])
        build_dataset_with_signal(actual_root / 'ds2', [2, 3])

        # Test with tilde path
        relative_to_home = actual_root.relative_to(home)
        tilde_path = Path('~') / relative_to_home

        result = load_all_datasets(tilde_path)

        assert len(result) == 4
        assert episode_ids(result[:]) == [0, 1, 2, 3]


# --- Edit log tests ---


def test_episode_meta_has_unique_uid(tmp_path):
    ds = build_dataset_with_signal(tmp_path / 'ds', [0, 1])
    uids = [ds[i].meta['uid'] for i in range(2)]
    assert all(isinstance(uid, str) and uid for uid in uids)
    assert uids[0] != uids[1]


def test_new_episode_carries_provided_uid(tmp_path):
    with LocalDatasetWriter(tmp_path / 'ds') as w:
        with w.new_episode(uid='source-uid') as ew:
            ew.set_static('id', 0)
    assert LocalDataset(tmp_path / 'ds')[0].meta['uid'] == 'source-uid'


def test_episode_without_stamped_uid_derives_one_from_created_ts(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    meta_path = root / '000000000000' / '000000000000' / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    del meta['uid']
    meta_path.write_text(json.dumps(meta), encoding='utf-8')

    uid = LocalDataset(root)[0].meta['uid']
    assert uid == f'ts-{meta["created_ts_ns"]}'

    edits.set_static(root, uid, {'verdict': 'success'})
    assert load_dataset(root)[0]['verdict'] == 'success'


def test_set_static_edit_applies_on_read(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0, 1])
    uid = ds[0].meta['uid']

    edits.set_static(root, uid, {'id': 100, 'verdict': 'success', 'blob': b'\x00\x01'})

    ds = load_dataset(root)
    assert ds[0]['id'] == 100
    assert ds[0]['verdict'] == 'success'
    assert ds[0]['blob'] == b'\x00\x01'
    assert ds[1].static == {'id': 1}
    # The recording itself stays untouched: the edit lives only in the log
    assert DiskEpisode(root / '000000000000' / '000000000000').static == {'id': 0}


def test_set_static_edit_last_write_wins(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0])
    uid = ds[0].meta['uid']

    edits.set_static(root, uid, {'verdict': 'fail', 'notes': 'first'})
    edits.set_static(root, uid, {'verdict': 'success'})

    ds = load_dataset(root)
    assert ds[0]['verdict'] == 'success'
    assert ds[0]['notes'] == 'first'


def test_set_static_edit_colliding_with_signal_raises(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0])
    edits.set_static(root, ds[0].meta['uid'], {'signal': 1})

    ds = load_dataset(root)
    with pytest.raises(ValueError, match='collide with signals'):
        _ = ds[0]


def test_set_static_edit_for_unknown_uid_is_inert(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    edits.set_static(root, 'no-such-uid', {'verdict': 'success'})

    ds = load_dataset(root)
    assert ds[0].static == {'id': 0}


def test_set_static_edit_rejects_invalid_values(tmp_path):
    with pytest.raises(ValueError, match='JSON-serializable'):
        edits.set_static(tmp_path, 'uid', {'bad': object()})
    with pytest.raises(ValueError, match='must be a mapping'):
        edits.set_static(tmp_path, 'uid', ['not', 'a', 'mapping'])
    with pytest.raises(ValueError, match='non-empty string'):
        edits.set_static(tmp_path, None, {'verdict': 'ok'})


def test_drop_edit_hides_episode(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0, 1, 2])
    edits.drop(root, ds[1].meta['uid'])

    assert episode_ids(load_dataset(root)[:]) == [0, 2]
    # The recording stays on disk; only the loaded view filters it
    assert len(LocalDataset(root)) == 3


def test_drop_edit_composes_with_set_static(tmp_path):
    root = tmp_path / 'ds'
    ds = build_dataset_with_signal(root, [0, 1])
    edits.set_static(root, ds[0].meta['uid'], {'verdict': 'success'})
    edits.drop(root, ds[1].meta['uid'])

    ds = load_dataset(root)
    assert len(ds) == 1
    assert ds[0]['verdict'] == 'success'


def test_drop_edit_for_unknown_uid_is_inert(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    edits.drop(root, 'no-such-uid')
    assert episode_ids(load_dataset(root)[:]) == [0]


def test_drop_edit_rejects_invalid_uid(tmp_path):
    with pytest.raises(ValueError, match='non-empty string'):
        edits.drop(tmp_path, '')


def test_corrupt_edit_record_raises(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    (root / edits.EDITS_FILE).write_text('{"op": "set_static", "v": 1, "ep": "x", "data": {', encoding='utf-8')
    with pytest.raises(ValueError, match='Corrupt edit record'):
        load_dataset(root)


def test_unsupported_edit_record_raises(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    (root / edits.EDITS_FILE).write_text('{"op": "trim", "v": 1, "ep": "x", "start": 0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported edit record'):
        load_dataset(root)


def test_edit_record_with_invalid_static_values_raises(tmp_path):
    root = tmp_path / 'ds'
    build_dataset_with_signal(root, [0])
    (root / edits.EDITS_FILE).write_text(
        '{"op": "set_static", "v": 1, "ep": "x", "data": {"maybe": null}}\n', encoding='utf-8'
    )
    with pytest.raises(ValueError, match='Invalid static values'):
        load_dataset(root)


def test_load_all_datasets_propagates_corrupt_edit_log(tmp_path):
    build_dataset_with_signal(tmp_path / 'ds1', [0, 1])
    build_dataset_with_signal(tmp_path / 'ds2', [2])
    (tmp_path / 'ds2' / edits.EDITS_FILE).write_text('garbage', encoding='utf-8')
    with pytest.raises(ValueError, match='Corrupt edit record'):
        load_all_datasets(tmp_path)


def test_load_all_datasets_ignores_edit_log_outside_datasets(tmp_path):
    build_dataset_with_signal(tmp_path / 'ds1', [0, 1])
    (tmp_path / edits.EDITS_FILE).write_text('garbage', encoding='utf-8')
    result = load_all_datasets(tmp_path)
    assert episode_ids(result[:]) == [0, 1]
