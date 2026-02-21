from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.utils import flatten_dict

from .base import Policy


class RemotePolicy(Policy):
    """Policy that forwards observations to a remote inference server.

    Images are resized before sending to reduce bandwidth. The server reports
    expected sizes via ``image_sizes`` in its metadata (see ``Codec.meta``).
    The ``resize`` parameter acts as a fallback when the server does not report
    sizes. Server-reported sizes always take precedence.

    ``horizon_sec`` truncates action chunks on the client side so only actions
    within the given time window are executed (e.g. ``horizon_sec=0.5`` keeps
    the first 0.5 s of each chunk). When ``None``, the full chunk is used as-is.
    """

    def __init__(
        self,
        host: str,
        port: int,
        resize: int | None = None,
        model_id: str | None = None,
        horizon_sec: float | None = None,
    ):
        self._client = InferenceClient(host, port)
        self.__session: InferenceSession | None = None
        self._resize = resize
        self._model_id = model_id
        self._horizon_sec = horizon_sec
        self._image_sizes: dict[str, tuple[int, int]] = {}
        self._default_image_size: tuple[int, int] | None = None

    def reset(self):
        """Resets the policy by starting a new session with the server."""
        self.close()
        self.__session = self._client.new_session(model_id=self._model_id)
        sizes = self.__session.metadata.get('image_sizes')
        if isinstance(sizes, dict):
            self._image_sizes = {k: tuple(v) for k, v in sizes.items()}
            self._default_image_size = None
        elif isinstance(sizes, tuple | list):
            self._default_image_size = tuple(sizes)
            self._image_sizes = {}
        else:
            self._default_image_size = None
            self._image_sizes = {}

    @property
    def _session(self) -> InferenceSession:
        if self.__session is None:
            self.reset()
        assert self.__session is not None
        return self.__session

    @staticmethod
    def _resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = image.shape[:2]
        if w == width and h == height:
            return image
        return np.array(PilImage.fromarray(image).resize((width, height), resample=PilImage.Resampling.BILINEAR))

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
                target = self._image_sizes.get(key, self._default_image_size)
                r = self._resize or 0
                tw, th = target or (r, r)
                if tw > 0 and th > 0:
                    h, w = value.shape[:2]
                    scale = min(1.0, tw / w, th / h)
                    value = self._resize_to(value, int(w * scale), int(h * scale))
            result[key] = value
        return result

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Forwards the observation to the remote server and returns the action.
        Uses client-side buffering if the server returns a chunk of actions.
        """
        # TODO: Remove horizon_sec from RemotePolicy — wrap with ActionHorizon codec instead
        # (requires splitting ActionTiming into ActionTimestamp + ActionHorizon first).
        actions = self._session.infer(self._prepare_obs(obs))
        if self._horizon_sec is not None and isinstance(actions, list):
            actions = [a for a in actions if a.get('timestamp', 0.0) < self._horizon_sec]
        return actions

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self._session.metadata})

    def close(self):
        if self.__session:
            self.__session.close()
            self.__session = None

    def __del__(self):
        self.close()
