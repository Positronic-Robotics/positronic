import numpy as np
import pytest

from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.vector import SimpleSignalWriter

pytest.importorskip('lance')

from positronic.vendors.lance.convert import _episode_row  # noqa: E402 — optional dependency checked above


def test_export_samples_only_the_selected_timeline(tmp_path):
    with LocalDatasetWriter(tmp_path / 'source') as writer, writer.new_episode(timeline='world') as episode:
        episode.set_static('task', 'pick')
        for i in range(3):
            timestamps = {'world': i * 1_000_000_000}
            episode.append('pose.joints', np.array([i], dtype=np.float32), timestamps)
            episode.append('camera', np.full((32, 32, 3), i * 30, dtype=np.uint8), timestamps)
        with SimpleSignalWriter(episode.path / 'foreign.parquet', timeline='wall') as signal_writer:
            signal_writer.append(9, {'wall': 1_000_000_000_000})
    source = LocalDataset(tmp_path / 'source')[0]
    assert isinstance(source, Episode)
    output = tmp_path / 'output'
    row = _episode_row(source, fps=1, output_dir=output, row_idx=0, timeline='world')
    assert row['task'] == 'pick'
    assert row['pose_joints'] == [[0], [1], [2]]
    assert row['trajectory_length'] == 3
    assert 'foreign' not in row
    assert row['camera_num_frames'] == 3
    assert row['camera_width'] == row['camera_height'] == 32
    assert (output / row['camera_uri']).is_file()
