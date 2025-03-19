import logging
import os
from pathlib import Path
from io import BytesIO

import cv2
import torch
import tqdm
import numpy as np
import imageio
import fire

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.compute_stats import compute_stats

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
        except Exception as e:
            try:
                print("Failed to decode video data. Trying to read from file.")
                with open('tmp.mp4', 'wb') as f:
                    f.write(array.numpy().tobytes())
                return torch.from_numpy(np.stack(imageio.mimread('tmp.mp4')))
            except Exception as e:
                raise ValueError(f"Failed to decode video data: {str(e)}")
            finally:
                os.remove('tmp.mp4')


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


def _start_from_zero(timestamp: torch.Tensor):
    return timestamp - timestamp[0]


@ir.config(
    fps=30,
    video=True,
    run_compute_stats=True,
    start_from_zero=True,
    timestamp_units='ns',
    state_encoder=positronic.cfg.inference.state.end_effector,
    action_encoder=positronic.cfg.inference.action.umi_relative,
)
def convert_to_lerobot_dataset(
    input_dir: str,
    output_dir: str,
    fps: int,
    video: bool,
    run_compute_stats: bool,
    start_from_zero: bool,
    timestamp_units: str,
    state_encoder: StateEncoder,
    action_encoder: ActionDecoder,
):  # noqa: C901  Function is too complex
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    # Process each episode file
    episode_files = sorted([f for f in input_dir.glob('*.pt')])
    total_length = 0

    all_episodes_data = []

    for episode_idx, episode_file in enumerate(tqdm.tqdm(episode_files, desc="Processing episodes")):
        episode_data = torch.load(episode_file)

        for key in episode_data.keys():
            # TODO: come up with a better way to determine if the data is a video (X2 !!!)
            if key.startswith('image.') or key.endswith('.image') and len(episode_data[key].shape) == 1:
                episode_data[key] = _decode_video_from_array(episode_data[key])

        ep_dict = {}
        obs = state_encoder.encode_episode(episode_data)

        # Process images
        for key in obs:
            if not key.startswith('observation.images.'):
                continue

            side = key.split('.')[-1]
            images = obs[key].numpy()

            video_filename = f"episode_{episode_idx:04d}_{side}.mp4"
            video_path = output_dir / "videos" / video_filename

            # Save frames as temporary PNG files
            temp_dir = output_dir / "temp_frames" / f"episode_{episode_idx:04d}_{side}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(tqdm.tqdm(images, desc=f"Saving frames for {side}")):
                img_path = temp_dir / f"frame_{i:06d}.png"
                if img.shape[0] == 3:  # If RGB image in CHW format
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.shape[0] == 1:  # If depth image in CHW format
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                cv2.imwrite(str(img_path), img)

            encode_video_frames(temp_dir, video_path, fps, overwrite=True)  # Encode video using ffmpeg

            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
            ep_dict[f"observation.images.{side}"] = [
                {"path": f"videos/{video_filename}", "timestamp": i / fps} for i in range(len(images))
            ]

        num_frames = len(episode_data['image_timestamp'])
        ep_dict['observation.state'] = obs['observation.state']

        # Concatenate all the data as specified in the config
        ep_dict['action'] = action_encoder.encode_episode(episode_data)

        ep_dict["episode_index"] = torch.tensor([episode_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        timestamp = episode_data['image_timestamp'].clone()

        if start_from_zero:
            timestamp = _start_from_zero(timestamp)

        ep_dict["timestamp"] = convert_to_seconds(timestamp_units, timestamp)
        total_length += ep_dict["timestamp"][-1] - ep_dict["timestamp"][0]

        done = torch.zeros(num_frames, dtype=torch.bool)
        # done[-1] = True
        ep_dict["next.done"] = done

        all_episodes_data.append(ep_dict)

        # Save individual episode
        ep_path = output_dir / "episodes" / f"episode_{episode_idx}.pth"
        torch.save(ep_dict, ep_path)

    # Concatenate all episodes
    data_dict = concatenate_episodes(all_episodes_data)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    # Create HuggingFace dataset
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    # Create LeRobotDataset
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="local/dataset",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=output_dir / "videos",
    )

    if run_compute_stats:
        logging.info("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset, batch_size=4, max_num_samples=10_000)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    # Save dataset components
    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that can't be saved
    hf_dataset.save_to_disk(str(output_dir / "train"))

    meta_data_dir = output_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    print(f"Dataset converted and saved to {output_dir}")
    print(f"Total length of the dataset: {total_length} seconds")


if __name__ == "__main__":
    fire.Fire(convert_to_lerobot_dataset.override_and_instantiate)
