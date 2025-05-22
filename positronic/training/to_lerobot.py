from pathlib import Path
from io import BytesIO
import tempfile

import torch
import tqdm
import numpy as np
import imageio
import fire

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import ironic as ir
import positronic.cfg.inference.action
import positronic.cfg.inference.state
from positronic.inference.action import ActionDecoder
from positronic.inference.state import StateEncoder


def _decode_video_from_array(array: torch.Tensor) -> torch.Tensor:
    """
    Decodes array with encoded video bytes into a video.

    Args:
        array (torch.Tensor): Tensor containing encoded video bytes.

    Returns:
        torch.Tensor: Decoded video frames.

    Raises:
        ValueError: If the video data cannot be decoded.
    """
    with BytesIO() as buffer:
        buffer.write(array.numpy().tobytes())
        buffer.seek(0)
        try:
            with imageio.get_reader(buffer, format='mp4') as reader:
                return torch.from_numpy(np.stack([frame for frame in reader]))
        except Exception:
            try:
                print("Failed to decode video data. Trying to read from file.")
                with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
                    tmp_file.write(array.numpy().tobytes())
                    return torch.from_numpy(np.stack(imageio.mimread(tmp_file.name)))
            except Exception as e:
                raise ValueError(f"Failed to decode video data: {str(e)}")


def convert_to_seconds(timestamp_units: str, timestamp: torch.Tensor):
    if timestamp_units == 'ns':
        return timestamp / 1e9
    elif timestamp_units == 'us':
        return timestamp / 1e6
    elif timestamp_units == 'ms':
        return timestamp / 1e3
    elif timestamp_units == 's':
        return timestamp
    else:
        raise ValueError(f"Unknown timestamp units: {timestamp_units}")


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
    def __init__(self, episode_files: list[Path], state_encoder: StateEncoder, action_encoder: ActionDecoder):
        self.episode_files = episode_files
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> dict:
        episode_file = self.episode_files[idx]
        episode_data = torch.load(episode_file)
        for key in episode_data.keys():
            # TODO: come up with a better way to determine if the data is a video (X2 !!!)
            if key.startswith('image.') or key.endswith('.image') and len(episode_data[key].shape) == 1:
                episode_data[key] = _decode_video_from_array(episode_data[key])
        obs = self.state_encoder.encode_episode(episode_data)
        ep_dict = {**obs}

        # Concatenate all the data as specified in the config
        ep_dict['action'] = self.action_encoder.encode_episode(episode_data)

        return ep_dict


def append_data_to_dataset(
    dataset: LeRobotDataset,
    input_dir: Path,
    state_encoder: StateEncoder,
    action_encoder: ActionDecoder,
    task: str,
    num_workers: int = 16,
):
    dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    episode_files = sorted([f for f in input_dir.glob('*.pt')])
    total_length = 0

    episode_dataset = EpisodeDictDataset(episode_files, state_encoder, action_encoder)
    dataloader = torch.utils.data.DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
        num_frames = len(ep_dict['action'])
        total_length += num_frames * 1 / dataset.fps

        for i in range(num_frames):
            frame = {"task": task}
            for key in ep_dict.keys():
                frame[key] = ep_dict[key][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        dataset.encode_episode_videos(episode_idx)
    print(f"Total length of the dataset: {seconds_to_str(total_length)}")


@ir.config(
    fps=30,
    video=True,
    state_encoder=positronic.cfg.inference.state.end_effector,
    action_encoder=positronic.cfg.inference.action.umi_relative,
    task="pick plate from the table and place it into the dishwasher",
)
def convert_to_lerobot_dataset(
    input_dir: str,
    output_dir: str,
    fps: int,
    video: bool,
    state_encoder: StateEncoder,
    action_encoder: ActionDecoder,
    task: str,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    features = {
        **state_encoder.get_features(),
        **action_encoder.get_features(),
    }

    dataset = LeRobotDataset.create(
        repo_id='local',
        fps=fps,
        root=output_dir,
        use_videos=video,
        features=features,
        image_writer_threads=32,
    )

    append_data_to_dataset(
        dataset=dataset,
        input_dir=input_dir,
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        task=task,
    )
    print(f"Dataset converted and saved to {output_dir}")


@ir.config(
    state_encoder=positronic.cfg.inference.state.end_effector,
    action_encoder=positronic.cfg.inference.action.umi_relative,
    task="pick plate from the table and place it into the dishwasher",
)
def append_data_to_lerobot_dataset(
    dataset_dir: str,
    input_dir: Path,
    state_encoder: StateEncoder,
    action_encoder: ActionDecoder,
    task: str,
):
    dataset = LeRobotDataset(
        repo_id='local',
        root=dataset_dir,
    )

    append_data_to_dataset(
        dataset=dataset,
        input_dir=Path(input_dir),
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        task=task,
    )
    print(f"Dataset extended with {input_dir} and saved to {dataset_dir}")


if __name__ == "__main__":
    fire.Fire({
        'convert': convert_to_lerobot_dataset.override_and_instantiate,
        'append': append_data_to_lerobot_dataset.override_and_instantiate,
    })
