"""
This utility converts Positronic datasets into LeRobot format. Now that
`lerobot` ships with the Positronic training dependencies, the easiest way to
run the tool is from the project environment (virtualenv or `uv run`).

Examples:
- Convert to a new LeRobot dataset
  positronic-to-lerobot convert \
    --dataset.path=/path/to/local_dataset \
    --output_dir=/path/to/lerobot_ds \
    --fps=30

- Append to an existing LeRobot dataset
  positronic-to-lerobot append \
    --output_dir=/path/to/lerobot_ds \
    --dataset.path=/path/to/local_dataset \
    --fps=30
"""

import json
import resource  # This will fail on Windows, as this library is Unix only, but we don't support Windows anyway
from collections.abc import Sequence as AbcSequence

import configuronic as cfn
import numpy as np
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import positronic.cfg.dataset
import positronic.utils.s3 as pos3
from positronic.dataset import Dataset


def _raise_fd_limit(min_soft_limit: int = 4096) -> None:
    """Increase soft RLIMIT_NOFILE to avoid LeRobot hitting macOS defaults."""
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
    """
    This dataset is used to load the episode data from the file and encode it into a dictionary.
    """

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


# This function needs to be serialisable for PyTorch DataLoader
def _collate_fn(x):
    return x[0]


def append_data_to_dataset(lr_dataset: LeRobotDataset, p_dataset: Dataset, fps, task=None, num_workers=16):
    _raise_fd_limit()
    lr_dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    total_length_sec = 0

    episode_dataset = EpisodeDictDataset(p_dataset, fps=fps)
    dataloader = torch.utils.data.DataLoader(
        episode_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn
    )

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc='Processing episodes')):
        num_frames = len(ep_dict['action'])
        total_length_sec += num_frames * 1 / lr_dataset.fps

        for i in range(num_frames):
            frame = {}
            for key, value in ep_dict.items():
                frame[key] = ep_dict[key]
                if isinstance(value, AbcSequence | np.ndarray) and len(value) == num_frames:
                    frame[key] = frame[key][i]

            ep_task = task
            if task is None:
                ep_task = frame.get('task', '')

            frame.pop('task', None)
            lr_dataset.add_frame(frame, task=ep_task or '')

        lr_dataset.save_episode()
        lr_dataset.encode_episode_videos(episode_idx)
    print(f'Total length of the dataset: {seconds_to_str(total_length_sec)}')


@cfn.config(video=True, dataset=positronic.cfg.dataset.transformed)
def convert_to_lerobot_dataset(output_dir: str, fps: int, video: bool, dataset: Dataset, task=None):
    output_dir = pos3.upload(output_dir, interval=None, sync_on_error=False)
    assert dataset.meta['lerobot_features'] is not None, "dataset.meta['lerobot_features'] is required"
    lr_dataset = LeRobotDataset.create(
        repo_id='local',
        fps=fps,
        root=output_dir,
        use_videos=video,
        features=dataset.meta['lerobot_features'],
        image_writer_threads=32,
    )
    if 'gr00t_modality' in dataset.meta:
        modality = dataset.meta.get('gr00t_modality')
        if modality is not None:
            modality['annotation'] = {'language.language_instruction': {'original_key': 'task_index'}}
            modality_path = output_dir / 'meta' / 'modality.json'
            with modality_path.open('w', encoding='utf-8') as f:
                json.dump(modality, f, indent=2)

    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task, fps=fps)
    print(f'Dataset converted and saved to {output_dir}')


@cfn.config(dataset=positronic.cfg.dataset.transformed)
def append_data_to_lerobot_dataset(output_dir: str, dataset: Dataset, fps: int, task=None):
    output_dir = pos3.upload(output_dir, interval=None, sync_on_error=False)
    lr_dataset = LeRobotDataset(repo_id='local', root=output_dir)

    lr_modality_path = output_dir / 'meta' / 'modality.json'
    ds_modality = dataset.meta.get('gr00t_modality', None)
    if ds_modality is not None:
        ds_modality['annotation'] = {'language.language_instruction': {'original_key': 'task_index'}}
    if lr_modality_path.exists():
        with lr_modality_path.open(encoding='utf-8') as f:
            lr_modality = json.load(f)
        if ds_modality is None or lr_modality != ds_modality:
            raise ValueError(
                "Mismatch in 'gr00t_modality':"
                " Existing LeRobot dataset modality and dataset.meta['gr00t_modality']"
                ' must both exist and be equal, or be absent from both.'
            )
    elif ds_modality is not None:
        # If dataset has modality but lerobot dataset doesn't, this is an error
        raise ValueError("'gr00t_modality' exists in dataset.meta but not in the destination LeRobot dataset.")

    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task, fps=fps)
    print(f'Dataset extended and saved to {output_dir}')


@pos3.with_mirror()
def _internal_main():
    cfn.cli({'convert': convert_to_lerobot_dataset, 'append': append_data_to_lerobot_dataset})


if __name__ == '__main__':
    _internal_main()
