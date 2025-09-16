# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot>=0.3.3",
#     "torch",
#     "tqdm",
#     "configuronic",
#     "numpy",
#     "scipy",
# ]
# ///
"""
PEP 723 standalone script
-------------------------

This script is intentionally decoupled from the core library’s dependencies so
that `lerobot` does not become a hard requirement of `positronic`.

How to run (recommended):
- Use uv to run the script in an isolated env that also installs the local
  project and its dependencies, while adding only the script-specific deps
  (like `lerobot`) as declared above.

Examples:
- Convert to a new LeRobot dataset
  uv run --with-editable . -s positronic/training/to_lerobot.py \
    convert --input-dir /path/to/local_dataset --output-dir /path/to/lerobot_ds

- Append to an existing LeRobot dataset
  uv run --with-editable . -s positronic/training/to_lerobot.py \
    append --dataset-dir /path/to/lerobot_ds --input-dir /path/to/local_dataset

Notes:
- The `--with-editable .` flag ensures your local `positronic` package and its
  dependencies are available inside the ephemeral uv environment without
  making `lerobot` a core dependency of the project.
- Run `uv --version` to ensure uv is installed. See: https://docs.astral.sh/uv/
"""
from collections.abc import Sequence as AbcSequence
from pathlib import Path

import numpy as np
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import configuronic as cfn
from positronic.cfg.policy import action, observation
from positronic.dataset import transforms, Dataset
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Kind
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder


def seconds_to_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


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


def append_data_to_dataset(lr_dataset: LeRobotDataset,
                           p_dataset: Dataset,
                           task: str | None = None,
                           num_workers: int = 16,
                           fps: int = 30):
    lr_dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    total_length_sec = 0

    episode_dataset = EpisodeDictDataset(p_dataset, fps=fps)
    dataloader = torch.utils.data.DataLoader(episode_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=_collate_fn)

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
        num_frames = len(ep_dict['action'])
        total_length_sec += num_frames * 1 / lr_dataset.fps

        for i in range(num_frames):
            ep_task = task
            if task is None and 'task' in ep_dict:
                ep_task = ep_dict['task']

            frame = {}
            for key, value in ep_dict.items():
                frame[key] = ep_dict[key]
                if isinstance(value, AbcSequence | np.ndarray) and len(value) == num_frames:
                    frame[key] = frame[key][i]

            lr_dataset.add_frame(frame, task=ep_task or '')

        lr_dataset.save_episode()
        lr_dataset.encode_episode_videos(episode_idx)
    print(f"Total length of the dataset: {seconds_to_str(total_length_sec)}")


def _extract_features(dataset: Dataset):
    """Extract feature metadata from dataset signals for LeRobot dataset creation."""
    features = {}
    for name, meta in dataset.signals_meta.items():
        feature = {'shape': meta.shape}
        if meta.names is not None:
            feature['names'] = meta.names

        if meta.kind == Kind.IMAGE:
            feature['dtype'] = 'video'
        else:
            feature['dtype'] = str(meta.dtype)

        features[name] = feature
    return features


@cfn.config(fps=30,
            video=True,
            state_encoder=observation.franka_mujoco_stackcubes,
            action_encoder=action.absolute_position,
            task="pick plate from the table and place it into the dishwasher")
def convert_to_lerobot_dataset(input_dir: str, output_dir: str, fps: int, video: bool,
                               state_encoder: ObservationEncoder, action_encoder: ActionDecoder, task: str):
    dataset = LocalDataset(Path(input_dir))
    dataset = transforms.TransformedDataset(dataset, state_encoder, action_encoder)

    lr_dataset = LeRobotDataset.create(repo_id='local',
                                       fps=fps,
                                       root=Path(output_dir),
                                       use_videos=video,
                                       features=_extract_features(dataset),
                                       image_writer_threads=32)

    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task)
    print(f"Dataset converted and saved to {output_dir}")


@cfn.config(state_encoder=observation.franka_mujoco_stackcubes,
            action_encoder=action.absolute_position,
            task="pick plate from the table and place it into the dishwasher")
def append_data_to_lerobot_dataset(dataset_dir: str, input_dir: Path, state_encoder: ObservationEncoder,
                                   action_encoder: ActionDecoder, task: str):
    lr_dataset = LeRobotDataset(repo_id='local', root=dataset_dir)

    dataset = LocalDataset(Path(input_dir))
    dataset = transforms.TransformedDataset(dataset, state_encoder, action_encoder)
    append_data_to_dataset(lr_dataset=lr_dataset, p_dataset=dataset, task=task)
    print(f"Dataset extended with {input_dir} and saved to {dataset_dir}")


if __name__ == "__main__":
    cfn.cli({'convert': convert_to_lerobot_dataset, 'append': append_data_to_lerobot_dataset})
