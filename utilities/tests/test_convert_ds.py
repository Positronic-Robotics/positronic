import numpy as np
import pyarrow.parquet as pq
import pytest

from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.transforms import Elementwise, TransformedDataset
from positronic.dataset.transforms.episode import Derive
from utilities.convert_ds import main


@pytest.mark.parametrize('data', [42, np.zeros((32, 32, 3), dtype=np.uint8)], ids=['scalar', 'image'])
@pytest.mark.parametrize('main_timeline, legacy', [(RECORDED_TIME, False), (RECORDED_TIME, True), ('world', False)])
def test_conversion_checks_main_timeline(tmp_path, data, main_timeline, legacy):
    source_root = tmp_path / 'source'
    destination = tmp_path / 'destination'
    with LocalDatasetWriter(source_root) as writer:
        with writer.new_episode(main_timeline=main_timeline) as episode:
            episode.append('signal', data, {main_timeline: 1000})
    if legacy:
        for path in source_root.rglob('*.parquet'):
            table = pq.read_table(path)
            pq.write_table(table.replace_schema_metadata(None), path)
    source = LocalDataset(source_root)
    if main_timeline != RECORDED_TIME:
        with pytest.raises(AssertionError, match="main timeline must be 'recorded', got 'world'"):
            main(output_path=str(destination), original_ds=source)
        assert len(LocalDataset(destination)) == 0
    else:
        main(output_path=str(destination), original_ds=source)
        signal = LocalDataset(destination)[0]['signal']
        assert signal.main_timeline == RECORDED_TIME
        assert list(signal.keys()) == [1000]
        np.testing.assert_array_equal(signal[0][0], data)


def test_conversion_rejects_custom_timeline_through_transforms(tmp_path):
    source_root = tmp_path / 'source'
    destination = tmp_path / 'destination'
    with LocalDatasetWriter(source_root) as writer:
        with writer.new_episode(main_timeline='world') as episode:
            episode.append('signal', 42, {'world': 1000})
    source = TransformedDataset(
        LocalDataset(source_root), Derive(signal=lambda episode: Elementwise(episode['signal'][:], np.asarray))
    )
    with pytest.raises(AssertionError, match="main timeline must be 'recorded', got 'world'"):
        main(output_path=str(destination), original_ds=source)
    assert len(LocalDataset(destination)) == 0
