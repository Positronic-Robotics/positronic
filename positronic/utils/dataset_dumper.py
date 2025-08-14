from collections import defaultdict
import os
from io import BytesIO

import numpy as np
import torch
import imageio.v2 as imageio


class SerialDumper:

    def __init__(
            self,
            directory: str,
            video_fps: int | None = None,
            codec: str = 'libx264',
            video_pixel_format: str = 'yuv444p'
    ):
        """
        Dumps serial data to a directory as torch tensors.
        Args:
            directory: Directory to save the data to. Episodes will be named sequentially.
            video_fps: Frames per second for video recording
            codec: Video codec to use (e.g., 'libx264', 'libx265')
            video_quality: Quality setting for video (1-10, higher is better)
            video_pixel_format: Pixel format for video (e.g., 'yuv420p', 'yuv444p')
        Example:
            >>> dumper = SerialDumper("data")
            >>> dumper.start_episode()
            >>> dumper.write({"position": np.array([1, 2, 3]), "time": 0.1})
            >>> dumper.write({"position": np.array([4, 5, 6]), "time": 0.2})
            >>> dumper.end_episode(metadata={"robot_type": "franka"})
        """
        self.directory = directory
        self.video_fps = video_fps
        self.codec = codec
        self.video_pixel_format = video_pixel_format
        os.makedirs(self.directory, exist_ok=True)
        self.data = defaultdict(list)
        self.video_buffers = {}
        self.video_writers = {}

        episode_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        if episode_files:
            episode_numbers = [int(f.split('.')[0]) for f in episode_files]
            self.episode_count = max(episode_numbers)
        else:
            self.episode_count = 0

    def start_episode(self):
        self.episode_count += 1

    def end_episode(self, metadata: dict = None):
        self.dump_episode(metadata=metadata)
        self.data = defaultdict(list)

    def write(self, data: dict | None = None, video_frames: dict | None = None):
        video_frames = video_frames or {}
        data = data or {}

        for k, v in video_frames.items():
            # import pdb; pdb.set_trace()
            assert self.video_fps is not None, "Video fps is not set. Please set it using the constructor."
            if k not in self.video_buffers:
                self.video_buffers[k] = BytesIO()
                self.video_writers[k] = imageio.get_writer(self.video_buffers[k],
                                                           fps=self.video_fps,
                                                           format='mp4',
                                                           codec=self.codec,
                                                           ffmpeg_params=[
                                                               '-vf', 'scale=1280:720',
                                                               '-preset', 'veryfast',
                                                           ])

            self.video_writers[k].append_data(v)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if len(self.data[k]) > 0:
                    assert self.data[k][-1].ctypes.data != v.ctypes.data, (
                        f"Appending the same np.ndarray to {k}. Make a copy of the array.")
                self.data[k].append(v)
            elif isinstance(v, torch.Tensor):
                if len(self.data[k]) > 0:
                    assert self.data[k][-1].data_ptr() != v.data_ptr(), (
                        f"Appending the same torch.tensor to {k}. Make a copy of the tensor.")
                self.data[k].append(v)
            elif isinstance(v, list):
                self.data[k].append(v.copy())
            elif isinstance(v, (int, float, str, np.number)):
                self.data[k].append(v)
            else:
                print(f"Appending {k} of type {type(v)}. Please check if you need to make a copy.")
                self.data[k].append(v)

    def dump_episode(self, metadata: dict = None):
        # Transform everything to torch tensors
        n_frames, tensor_key = None, None

        for k, v in self.data.items():
            try:
                self.data[k] = torch.from_numpy(np.array(v))
            except Exception as e:
                print(f"Error converting {k} to torch tensor: {e}")
                raise e

            if n_frames is None:
                n_frames = len(self.data[k])
                tensor_key = k
            else:
                # TODO: It's currently assumed, but in the future we might want to support different length tensors.
                assert len(self.data[k]) == n_frames, (
                    "All tensors must have the same length. "
                    f"Got {len(self.data[k])} and {n_frames} for {k} and {tensor_key}.")

        for k, v in self.video_buffers.items():
            self.video_writers[k].close()
            self.data[k] = torch.from_numpy(np.frombuffer(v.getvalue(), dtype=np.uint8).copy())
            self.video_buffers[k].close()

        self.video_buffers = {}
        self.video_writers = {}

        if metadata is not None:
            for k in metadata.keys():
                assert k not in self.data, f"Metadata key {k} intersects with data key."
                if isinstance(metadata[k], np.ndarray):
                    metadata[k] = torch.from_numpy(metadata[k])

            self.data.update(metadata)

        fname = f"{self.directory}/{str(self.episode_count).zfill(3)}.pt"
        torch.save(dict(self.data), fname)
        print(f"Episode {self.episode_count} saved to {fname} with {n_frames} frames")
