import json
import os

import numpy as np
import pytest

from pimm.time import EMITTED_WORLD, RECEIVED_WORLD
from positronic import keys
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.time import HARNESS_WORLD, Time, TimeBounds
from positronic.dataset.transforms import TransformedDataset
from positronic.policy.codecs import ACTION, LEROBOT_FEATURES, AbsoluteJointsAction, ObservationCodec

lerobot = pytest.importorskip('lerobot')
if not hasattr(lerobot, '__version__') or lerobot.__version__ < '0.4':
    pytest.skip('Requires lerobot >= 0.4', allow_module_level=True)

os.environ['HF_HUB_OFFLINE'] = '1'

import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from positronic.vendors.lerobot.to_lerobot import EpisodeDictDataset, append_data_to_dataset  # noqa: E402


def test_export_keeps_actions_and_observations_aligned_on_harness_time(tmp_path):
    source = tmp_path / 'source'
    with LocalDatasetWriter(source) as dataset, dataset.new_episode() as writer:
        for i in range(1, 4):
            received = i * 100_000_000
            emitted = received + 10_000_000
            writer.append(
                keys.JOINTS,
                np.array([i]),
                Time(**{EMITTED_WORLD: received - 10_000_000, RECEIVED_WORLD: received, HARNESS_WORLD: received}),
            )
            time = Time(**{EMITTED_WORLD: emitted, HARNESS_WORLD: emitted})
            writer.append(keys.TARGET_JOINTS, np.array([i * 10]), time)
            writer.append(keys.TARGET_GRIP, i / 10, time)
    codec = ObservationCodec(state={'state': {keys.JOINTS: 1}}, images={}) & AbsoluteJointsAction(
        keys.TARGET_JOINTS, keys.TARGET_GRIP, num_joints=1
    )
    encoder = codec.training_encoder
    dataset = TransformedDataset(LocalDataset(source), encoder)
    rows = EpisodeDictDataset(dataset, fps=20)[0]
    np.testing.assert_array_equal(rows['state'], [[1], [1], [2], [2]])
    np.testing.assert_allclose(rows[ACTION], [[10, 0.1], [10, 0.1], [20, 0.2], [20, 0.2]])

    output = tmp_path / 'lerobot'
    target = LeRobotDataset.create(
        repo_id='local', fps=20, root=output, use_videos=False, features=encoder.meta[LEROBOT_FEATURES]
    )
    append_data_to_dataset(target, dataset, fps=20, task='test task', num_workers=0)
    loaded = LeRobotDataset(repo_id='local', root=output)
    assert len(loaded) == 4
    for index in range(4):
        np.testing.assert_allclose(loaded[index][ACTION].numpy(), rows[ACTION][index])
        np.testing.assert_array_equal(loaded[index]['state'].numpy(), rows['state'][index])


class _MockTimeIndex:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, timestamps):
        num_frames = len(timestamps)
        result = {}
        for key, val in self._data.items():
            if isinstance(val, np.ndarray) and val.shape[0] >= num_frames:
                result[key] = val[:num_frames]
            else:
                result[key] = val
        return result


class _MockEpisode:
    timelines = (RECORDED_TIME,)

    def __init__(self, num_frames, fps):
        self._bounds = (Time(**{RECORDED_TIME: 0}), Time(**{RECORDED_TIME: int(num_frames * 1e9 / fps)}))
        data = {
            'observation.state': np.random.randn(num_frames, 8).astype(np.float32),
            'action': np.random.randn(num_frames, 8).astype(np.float32),
        }
        self.time = _MockTimeIndex(data)

    def bounds(self, timelines):
        return TimeBounds(self._bounds[0][timelines], self._bounds[1][timelines])


class _MockDataset(torch.utils.data.Dataset):
    def __init__(self, num_episodes=2, num_frames=5, fps=15):
        self.episodes = [_MockEpisode(num_frames, fps) for _ in range(num_episodes)]
        self.meta = {
            'action_fps': fps,
            'lerobot_features': {
                'observation.state': {'shape': (8,), 'dtype': 'float32'},
                'action': {'shape': (8,), 'dtype': 'float32'},
            },
        }
        self.fps = fps

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def test_convert_to_lerobot_e2e(tmp_path):
    """E2e: mock positronic dataset -> convert to v3.0 -> verify structure."""
    num_episodes = 2
    num_frames = 5
    fps = 15
    output_dir = tmp_path / 'lerobot_output'

    mock_dataset = _MockDataset(num_episodes=num_episodes, num_frames=num_frames, fps=fps)

    lr_dataset = LeRobotDataset.create(
        repo_id='local', fps=fps, root=output_dir, use_videos=False, features=mock_dataset.meta['lerobot_features']
    )

    append_data_to_dataset(lr_dataset, mock_dataset, fps=fps, task='test task', num_workers=0)

    # Verify meta/info.json
    info_path = output_dir / 'meta' / 'info.json'
    assert info_path.exists(), f'meta/info.json not found at {info_path}'

    with info_path.open() as f:
        info = json.load(f)

    assert info['codebase_version'].startswith('v'), f'Unexpected codebase_version: {info["codebase_version"]}'
    assert info['fps'] == fps
    assert info['total_episodes'] == num_episodes

    # Verify data parquet files exist
    data_dir = output_dir / 'data'
    assert data_dir.exists(), f'data/ directory not found at {data_dir}'
    parquet_files = list(data_dir.rglob('*.parquet'))
    assert len(parquet_files) > 0, 'No parquet files found'

    # Verify by loading with LeRobotDataset
    loaded = LeRobotDataset(repo_id='local', root=output_dir)
    assert loaded.num_episodes == num_episodes
    assert len(loaded) == num_episodes * num_frames
