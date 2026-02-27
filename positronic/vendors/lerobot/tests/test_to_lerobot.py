import json
import os

import numpy as np
import pytest

lerobot = pytest.importorskip('lerobot')
if not hasattr(lerobot, '__version__') or lerobot.__version__ < '0.4':
    pytest.skip('Requires lerobot >= 0.4', allow_module_level=True)

os.environ['HF_HUB_OFFLINE'] = '1'

import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from positronic.vendors.lerobot.to_lerobot import append_data_to_dataset  # noqa: E402


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
    def __init__(self, num_frames, fps):
        self.start_ts = 0
        self.last_ts = int(num_frames * 1e9 / fps)
        data = {
            'observation.state': np.random.randn(num_frames, 8).astype(np.float32),
            'action': np.random.randn(num_frames, 8).astype(np.float32),
        }
        self.time = _MockTimeIndex(data)


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
