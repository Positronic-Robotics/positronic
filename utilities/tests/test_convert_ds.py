import numpy as np
import pyarrow.parquet as pq
import pytest

from positronic.dataset import Time
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import RECORDED_TIME, TIMELINE_METADATA_KEY
from positronic.dataset.transforms import Elementwise, TransformedDataset
from positronic.dataset.transforms.episode import Derive
from utilities.convert_ds import main


@pytest.mark.parametrize('data', [42, np.zeros((32, 32, 3), dtype=np.uint8)], ids=['scalar', 'image'])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('timeline', [RECORDED_TIME, 'world'])
def test_conversion_preserves_exposed_timelines(tmp_path, data, timeline, legacy):
    source_root = tmp_path / 'source'
    destination = tmp_path / 'destination'
    timestamps = Time(**{timeline: 1000}, wall=2000)
    with LocalDatasetWriter(source_root) as writer:
        with writer.new_episode() as episode:
            episode.append('signal', data, timestamps)
    if legacy:
        for path in source_root.rglob('*.parquet'):
            table = pq.read_table(path)
            main_column = 'ts_ns' if path.name.endswith('.frames.parquet') else 'timestamp'
            table = table.rename_columns([
                main_column if name == f'ts.{timeline}' else 'ts_ns.wall' if name == 'ts.wall' else name
                for name in table.column_names
            ])
            pq.write_table(table.replace_schema_metadata({TIMELINE_METADATA_KEY: timeline.encode()}), path)
    source = LocalDataset(source_root)
    main(output_path=str(destination), original_ds=source)
    signal = LocalDataset(destination)[0]['signal']
    expected = timestamps[(timeline,)] if legacy else timestamps
    assert signal.timelines == expected.timelines
    assert signal[0][1] == expected
    np.testing.assert_array_equal(signal[0][0], data)


def test_conversion_preserves_timelines_through_transforms(tmp_path):
    source_root = tmp_path / 'source'
    destination = tmp_path / 'destination'
    with LocalDatasetWriter(source_root) as writer:
        with writer.new_episode() as episode:
            episode.append('signal', 42, Time(world=1000, tick=1))
    source = TransformedDataset(
        LocalDataset(source_root), Derive(signal=lambda episode: Elementwise(episode['signal'][:], np.asarray))
    )
    main(output_path=str(destination), original_ds=source)
    assert LocalDataset(destination)[0]['signal'][0] == (42, Time(world=1000, tick=1))
