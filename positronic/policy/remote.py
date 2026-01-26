from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.utils import flatten_dict

from .base import Policy


class RemotePolicy(Policy):
    """
    A policy that forwards observations to a remote inference server using the
    Positronic Inference Protocol.
    """

    def __init__(self, host: str, port: int, resize: int | None = None, model_id: str | None = None):
        self._client = InferenceClient(host, port)
        self.__session: InferenceSession | None = None
        self._resize = resize
        self._model_id = model_id

    def reset(self):
        """
        Resets the policy by starting a new session with the server.
        """
        self.close()
        self.__session = self._client.new_session(model_id=self._model_id)

    @property
    def _session(self) -> InferenceSession:
        if self.__session is None:
            self.reset()
        assert self.__session is not None
        return self.__session

    @staticmethod
    def _resize_if_needed(image: np.ndarray, max_resolution: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(1, max_resolution / max(width, height))
        max_width, max_height = int(width * scale), int(height * scale)

        # Downscale if needed
        if width != max_width or height != max_height:
            new_size = max_width, max_height
            return np.array(PilImage.fromarray(image).resize(new_size, resample=PilImage.Resampling.BILINEAR))
        return image

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._resize is None:
            return obs

        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
                result[key] = self._resize_if_needed(value, self._resize)
            else:
                result[key] = value
        return result

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Forwards the observation to the remote server and returns the action.
        Uses client-side buffering if the server returns a chunk of actions.
        """
        return self._session.infer(self._prepare_obs(obs))

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self._session.metadata})

    def close(self):
        if self.__session:
            self.__session.close()
            self.__session = None

    def __del__(self):
        self.close()
