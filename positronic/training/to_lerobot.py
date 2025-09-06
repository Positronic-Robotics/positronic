# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot>=0.3",
#     "torch",
#     "tqdm",
#     "configuronic",
#     "numpy",
#     "scipy",
# ]
# ///
from collections.abc import Sequence as AbcSequence
from pathlib import Path

import numpy as np
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import configuronic as cfn
from positronic.cfg.inference import action, state
from positronic.dataset import transforms
from positronic.dataset.local_dataset import LocalDataset
from positronic.inference.action import ActionDecoder
from positronic.inference.state import ObservationEncoder


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

    def __init__(self, input_dir: Path, state_encoder: ObservationEncoder, action_encoder: ActionDecoder, fps: int):
        self.dataset = LocalDataset(input_dir)
        self.observation_encoder = state_encoder
        self.action_encoder = action_encoder
        self.fps = fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        episode = transforms.TransformEpisode(self.dataset[idx], self.observation_encoder, self.action_encoder)
        start, finish = episode.start_ts, episode.last_ts
        timestamps = np.arange(start, finish, 1e9 / self.fps, dtype=np.int64)
        return episode.time[timestamps]


# This function needs to be serialisable for PyTorch DataLoader
def _collate_fn(x):
    return x[0]


def append_data_to_dataset(dataset: LeRobotDataset,
                           input_dir: Path,
                           state_encoder: ObservationEncoder,
                           action_encoder: ActionDecoder,
                           task: str | None = None,
                           num_workers: int = 16,
                           fps: int = 30):
    dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    total_length_sec = 0

    episode_dataset = EpisodeDictDataset(input_dir, state_encoder, action_encoder, fps=fps)
    dataloader = torch.utils.data.DataLoader(episode_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=_collate_fn)

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
        num_frames = len(ep_dict['action'])
        total_length_sec += num_frames * 1 / dataset.fps

        for i in range(num_frames):
            ep_task = task
            if task is None and 'task' in ep_dict:
                ep_task = ep_dict['task']

            frame = {}
            for key, value in ep_dict.items():
                frame[key] = ep_dict[key]
                if isinstance(value, AbcSequence | np.ndarray) and len(value) == num_frames:
                    frame[key] = frame[key][i]

            dataset.add_frame(frame, task=ep_task or '')

        dataset.save_episode()
        dataset.encode_episode_videos(episode_idx)
    print(f"Total length of the dataset: {seconds_to_str(total_length_sec)}")


@cfn.config(fps=30,
            video=True,
            state_encoder=state.franka_mujoco_stackcubes,
            action_encoder=action.absolute_position,
            task="pick plate from the table and place it into the dishwasher")
def convert_to_lerobot_dataset(input_dir: str, output_dir: str, fps: int, video: bool,
                               state_encoder: ObservationEncoder, action_encoder: ActionDecoder, task: str):
    features = {**state_encoder.get_features(), **action_encoder.get_features()}

    dataset = LeRobotDataset.create(repo_id='local',
                                    fps=fps,
                                    root=Path(output_dir),
                                    use_videos=video,
                                    features=features,
                                    image_writer_threads=32)

    append_data_to_dataset(dataset=dataset,
                           input_dir=Path(input_dir),
                           state_encoder=state_encoder,
                           action_encoder=action_encoder,
                           task=task)
    print(f"Dataset converted and saved to {output_dir}")


@cfn.config(state_encoder=state.franka_mujoco_stackcubes,
            action_encoder=action.absolute_position,
            task="pick plate from the table and place it into the dishwasher")
def append_data_to_lerobot_dataset(dataset_dir: str, input_dir: Path, state_encoder: ObservationEncoder,
                                   action_encoder: ActionDecoder, task: str):
    dataset = LeRobotDataset(repo_id='local', root=dataset_dir)

    append_data_to_dataset(dataset=dataset,
                           input_dir=Path(input_dir),
                           state_encoder=state_encoder,
                           action_encoder=action_encoder,
                           task=task)
    print(f"Dataset extended with {input_dir} and saved to {dataset_dir}")


if __name__ == "__main__":
    cfn.cli({'convert': convert_to_lerobot_dataset, 'append': append_data_to_lerobot_dataset})
