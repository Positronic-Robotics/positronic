import os

import numpy as np
import pytest

from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.vector import SimpleSignalWriter
from positronic.policy.codec import ACTION

lerobot = pytest.importorskip('lerobot')
if lerobot.__version__ != '0.3.3':
    pytest.skip('Requires lerobot 0.3.3', allow_module_level=True)

os.environ['HF_HUB_OFFLINE'] = '1'

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402 — optional dependency checked above

from positronic.vendors.lerobot_0_3_3.to_lerobot import (  # noqa: E402 — optional dependency checked above
    append_data_to_dataset,
)


@pytest.mark.parametrize('timeline', [RECORDED_TIME, 'world'])
def test_export_samples_only_the_selected_timeline(tmp_path, timeline):
    with LocalDatasetWriter(tmp_path / 'source') as writer, writer.new_episode(timeline=timeline) as episode:
        for i in range(3):
            episode.append(ACTION, np.array([i], dtype=np.float32), {timeline: i * 1_000_000_000})
        with SimpleSignalWriter(episode.path / 'foreign.parquet', timeline='wall') as signal:
            signal.append(9, {'wall': 1_000_000_000_000})

    output_dir = tmp_path / 'lerobot'
    output = LeRobotDataset.create(
        repo_id='local',
        fps=1,
        root=output_dir,
        use_videos=False,
        features={ACTION: {'shape': (1,), 'dtype': 'float32'}},
    )
    append_data_to_dataset(
        output, LocalDataset(tmp_path / 'source'), fps=1, task='pick', num_workers=0, timeline=timeline
    )

    result = LeRobotDataset(repo_id='local', root=output_dir)
    assert result.num_episodes == 1
    assert result.num_frames == 2
    np.testing.assert_array_equal([result[i][ACTION].item() for i in range(len(result))], [0, 1])
    assert 'foreign' not in result.features
