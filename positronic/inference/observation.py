import abc

import cv2

from datasets.table import np
from positronic.dataset.core import Episode, Signal


class Transform(abc.ABC):
    @abc.abstractmethod
    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        pass


class ImageTransform(Transform):
    def __init__(self, input_key: str, output_key: str, resize: tuple[int, int]):
        self.input_key = input_key
        self.output_key = output_key
        self.resize = resize

    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        images = signal_dict[self.input_key].time[timestamps]
        resized = []

        for image, ts in images:
            image = cv2.resize(image, self.resize)
            resized.append(image)

        return {
            self.output_key: np.array(resized)
        }


class ToArrayTransform(Transform):
    def __init__(self, input_key: str, n_features: int, output_key: str | None = None):
        self.input_key = input_key
        self.output_key = output_key if output_key is not None else input_key
        self.n_features = n_features

    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        data = np.array([x[0] for x in signal_dict[self.input_key].time[timestamps]])
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if data.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {data.shape[1]}")

        return {
            self.output_key: data
        }


class ObservationEncoder:
    def __init__(self, transforms: list[Transform], fps: int):
        self.transforms = transforms
        self.fps = fps

    def episode_to_dict(self, episode: Episode, timestamps: np.ndarray) -> dict:
        observations = {}

        signal_dict = {key: episode[key] for key in episode.keys(static=False)}

        for transform in self.transforms:
            observations.update(transform.encode(signal_dict, timestamps))

        return observations
