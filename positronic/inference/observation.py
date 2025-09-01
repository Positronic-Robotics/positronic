import abc

import cv2

from datasets.table import np
from positronic.dataset.core import Episode, Signal
from PIL import Image

class Transform(abc.ABC):
    @abc.abstractmethod
    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        pass


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


class ImageTransform(Transform):
    def __init__(self, input_key: str, output_key: str, resize: tuple[int, int], pad: bool = False):
        self.input_key = input_key
        self.output_key = output_key
        self.resize = resize
        self.pad = pad

    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        images = signal_dict[self.input_key].time[timestamps]
        resized = []

        for image, ts in images:
            if self.pad:
                image = resize_with_pad(image, self.resize[0], self.resize[1])
            else:
                image = cv2.resize(image, self.resize)
            resized.append(image)

        return {
            self.output_key: np.array(resized)
        }


class ToArrayTransform(Transform):
    def __init__(self, input_key: str | list[str], n_features: int, output_key: str | None = None):
        self.input_key = input_key
        self.output_key = output_key if output_key is not None else input_key
        self.n_features = n_features

    def encode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> dict:
        if isinstance(self.input_key, str):
            data = np.array([x[0] for x in signal_dict[self.input_key].time[timestamps]])
        else:
            data = []
            for key in self.input_key:
                x = np.array([x[0] for x in signal_dict[key].time[timestamps]])
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                print(x.shape)
                data.append(x)
            data = np.concatenate(data, axis=1)

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
