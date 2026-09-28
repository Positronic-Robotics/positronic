import numpy as np
import pyarrow.parquet as pq
import pytest

from positronic.dataset.episode import META_CREATED_TS_NS, META_UID
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Derive, Group, Identity, Rename
from positronic.dataset.vector import SimpleSignalWriter
from utilities.convert_ds import main


@pytest.mark.parametrize('timeline', [RECORDED_TIME, 'world'])
def test_convert_materializes_transforms_and_preserves_encoded_signals(tmp_path, timeline):
    with LocalDatasetWriter(tmp_path / 'source') as writer, writer.new_episode(timeline=timeline) as episode:
        episode.set_static('task', 'pick')
        for i in range(2):
            timestamps = {timeline: 100 + i, 'tick': i}
            episode.append('raw', i, timestamps)
            episode.append('cam', np.full((64, 64, 3), i * 100, dtype=np.uint8), timestamps)
        with SimpleSignalWriter(episode.path / 'foreign.parquet', timeline='wall') as signal:
            signal.append(7, {'wall': 1000})
    source = LocalDataset(tmp_path / 'source')
    transformed = TransformedDataset(
        source,
        Group(
            Rename(raw_copy='raw', video_copy='cam', foreign_copy='foreign'),
            Derive(derived=lambda ep: ep['raw'][:]),
            Identity(select=['task']),
        ),
    )

    main(output_path=str(tmp_path / 'output'), original_ds=transformed)

    copied = LocalDataset(tmp_path / 'output')
    assert len(copied) == 1
    result = copied[0]
    assert result.static['task'] == 'pick'
    assert result.meta[META_UID] == source[0].meta[META_UID]
    assert result.meta[META_CREATED_TS_NS] == source[0].meta[META_CREATED_TS_NS]
    assert set(result.signals) == {'raw_copy', 'video_copy', 'foreign_copy', 'derived'}
    assert result['derived'].timeline == timeline
    assert list(result['derived']) == [(0, 100), (1, 101)]
    assert result['foreign_copy'].timeline == 'wall'
    assert list(result['foreign_copy']) == [(7, 1000)]
    assert result['raw_copy'].filepath.read_bytes() == source[0]['raw'].filepath.read_bytes()
    assert result['video_copy'].video_path.read_bytes() == source[0]['cam'].video_path.read_bytes()
    assert result['video_copy'].timeline == timeline
    assert pq.read_table(result['video_copy'].frames_index_path).equals(
        pq.read_table(source[0]['cam'].frames_index_path)
    )
    np.testing.assert_array_equal(result['video_copy'].values(), source[0]['cam'].values())
