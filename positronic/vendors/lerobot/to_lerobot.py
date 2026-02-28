"""
Convert Positronic datasets to LeRobot v3.0 format using lerobot 0.4.x API.

Examples:
- Convert to a new LeRobot dataset
  uv run --extra lerobot python -m positronic.vendors.lerobot.to_lerobot convert \\
    --dataset.path=/path/to/local_dataset \\
    --output_dir=/path/to/lerobot_ds

- Append to an existing LeRobot dataset
  uv run --extra lerobot python -m positronic.vendors.lerobot.to_lerobot append \\
    --output_dir=/path/to/lerobot_ds \\
    --dataset.path=/path/to/local_dataset
"""

import logging
import resource
from collections.abc import Sequence as AbcSequence

import configuronic as cfn
import numpy as np
import pos3
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from positronic import utils
from positronic.cfg.ds import apply_codec
from positronic.dataset import Dataset
from positronic.utils.logging import init_logging


def _raise_fd_limit(min_soft_limit: int = 4096) -> None:
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard_limit, max(soft_limit, min_soft_limit))

    if soft_limit < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard_limit))
        except (ValueError, OSError):
            pass


def seconds_to_str(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.2f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.2f}m'
    else:
        return f'{seconds / 3600:.2f}h'


class EpisodeDictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, fps: int):
        self.dataset = dataset
        self.fps = fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        episode = self.dataset[idx]
        start, finish = episode.start_ts, episode.last_ts
        timestamps = np.arange(start, finish, 1e9 / self.fps, dtype=np.int64)
        return episode.time[timestamps]


def _collate_fn(x):
    return x[0]


def append_data_to_dataset(lr_dataset: LeRobotDataset, p_dataset: Dataset, fps, task=None, num_workers=16):
    _raise_fd_limit()
    lr_dataset.start_image_writer(num_processes=num_workers)
    total_length_sec = 0

    episode_dataset = EpisodeDictDataset(p_dataset, fps=fps)
    dataloader = torch.utils.data.DataLoader(
        episode_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn
    )

    for _episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc='Processing episodes')):
        num_frames = len(ep_dict['action'])
        total_length_sec += num_frames * 1 / lr_dataset.fps

        for i in range(num_frames):
            frame = {}
            for key, value in ep_dict.items():
                frame[key] = ep_dict[key]
                if isinstance(value, AbcSequence | np.ndarray) and len(value) == num_frames:
                    frame[key] = frame[key][i]

            ep_task = task if task is not None else frame.get('task', '')
            frame['task'] = ep_task or ''
            lr_dataset.add_frame(frame)

        lr_dataset.save_episode()
    lr_dataset.finalize()
    logging.info(f'Total length of the dataset: {seconds_to_str(total_length_sec)}')


@cfn.config(video=True, dataset=apply_codec, fps=None)
def convert_to_lerobot_dataset(output_dir: str, fps: int | None, video: bool, dataset: Dataset, task=None):
    if fps is None:
        assert 'action_fps' in dataset.meta, "--fps not provided and dataset has no 'action_fps' metadata"
        fps = int(dataset.meta['action_fps'])
    output_dir = pos3.sync(output_dir, interval=None, sync_on_error=False)
    assert dataset.meta['lerobot_features'] is not None, "dataset.meta['lerobot_features'] is required"

    lr_dataset = LeRobotDataset.create(
        repo_id='local',
        fps=fps,
        root=output_dir,
        use_videos=video,
        features=dataset.meta['lerobot_features'],
        image_writer_threads=32,
    )
    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task, fps=fps)
    logging.info(f'Dataset converted and saved to {output_dir}')


@cfn.config(dataset=apply_codec, fps=None)
def append_data_to_lerobot_dataset(output_dir: str, dataset: Dataset, fps: int | None, task=None):
    if fps is None:
        assert 'action_fps' in dataset.meta, "--fps not provided and dataset has no 'action_fps' metadata"
        fps = int(dataset.meta['action_fps'])
    output_dir = pos3.sync(output_dir, interval=None, sync_on_error=False)
    lr_dataset = LeRobotDataset(repo_id='local', root=output_dir)

    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix='append_metadata')

    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task, fps=fps)
    logging.info(f'Dataset extended and saved to {output_dir}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'convert': convert_to_lerobot_dataset, 'append': append_data_to_lerobot_dataset})


if __name__ == '__main__':
    _internal_main()
