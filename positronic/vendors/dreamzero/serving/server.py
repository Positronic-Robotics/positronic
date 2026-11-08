"""Serve DreamZero with bounded observation history and periodic ordinary inference calls."""

import logging
import math
import os
from collections import deque
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import Any

import configuronic as cfn
import numpy as np
from positronic_model_server.serialization import check_jpeg_quality
from positronic_model_server.server import Model, ModelServer, Session
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import WebsocketWire

from . import recipe
from .keys import (
    ACTION_JOINTS,
    AUTH_TOKEN_ENV,
    CAMERAS,
    CHECKPOINT_PATH,
    GRIP,
    JOINTS,
    MODEL_SOURCE_ENV,
    OBS_TIME_NS,
    PROMPT,
    TYPE,
)
from .recipe import Mode

logger = logging.getLogger(__name__)


class DreamZeroSession(Session):
    """Keep a bounded sampled history; advance native video state only on prediction calls.

    Prediction deadlines share the first observation's clock. Missed deadlines use the latest
    supplied observation. History timestamps and targets map to the nearest sampling tick;
    missing ticks use the most recent older sample, or the earliest retained sample at startup.
    Observations older than or equal to the latest accepted timestamp return ``None`` without changing state.
    """

    def __init__(
        self,
        policy: Any,
        client_stack: dict,
        *,
        max_in_flight: int,
        sampling_period_sec: float,
        prediction_period_sec: float,
        history_offsets_sec: tuple[float, ...],
        width: int,
        height: int,
        release: Callable[[], None],
    ):
        super().__init__(client_stack, max_in_flight=max_in_flight)
        self._policy = policy
        self._period_ns = round(prediction_period_sec * 1e9)
        self._sampling_period_ns = sampling_period_sec * 1e9
        self._offsets_ns = tuple(round(offset * 1e9) for offset in history_offsets_sec)
        capacity = math.ceil(-self._offsets_ns[0] / (sampling_period_sec * 1e9)) + 2
        self._history: deque[tuple[int, dict[str, np.ndarray]]] = deque(maxlen=capacity)
        self._shape = (height, width, 3)
        self._origin_ns: int | None = None
        self._next_prediction_ns = 0
        self._release = release
        self._closed = False

    def infer(self, observation: Any) -> Any:
        if self._closed:
            raise RuntimeError('Session is closed')
        now_ns = observation[OBS_TIME_NS]
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError('obs_time_ns must be a nonnegative integer')
        if self._history and now_ns <= self._history[-1][0]:
            return None
        images = {name: np.asarray(observation[name]) for name in CAMERAS}
        if any(image.shape != self._shape or image.dtype != np.uint8 for image in images.values()):
            raise ValueError(f'Camera images must be uint8 with shape {self._shape}')
        self._history.append((now_ns, images))
        while len(self._history) > 1 and self._history[1][0] <= now_ns + self._offsets_ns[0]:
            self._history.popleft()
        first = self._origin_ns is None
        if first:
            self._origin_ns = now_ns
        elif now_ns < self._next_prediction_ns:
            return None
        assert self._origin_ns is not None
        offsets = (0,) if first else self._offsets_ns
        frames = []
        for offset in offsets:
            target_index = round((now_ns + offset - self._origin_ns) / self._sampling_period_ns)
            frame = next(
                (
                    images
                    for stamp, images in reversed(self._history)
                    if round((stamp - self._origin_ns) / self._sampling_period_ns) <= target_index
                ),
                self._history[0][1],
            )
            frames.append(frame)
        inputs = {name: np.stack([frame[name] for frame in frames]) for name in CAMERAS}
        inputs.update({name: observation[name] for name in (JOINTS, GRIP, PROMPT)})
        result = self._policy.predict(inputs)
        periods = (now_ns - self._origin_ns) // self._period_ns + 1
        self._next_prediction_ns = self._origin_ns + periods * self._period_ns
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._history.clear()
            self._release()


class DreamZeroModel(Model):
    """One active session owns the native autoregressive caches; weights stay loaded between sessions."""

    def __init__(
        self,
        policy: Any,
        *,
        action_horizon: int = 24,
        max_in_flight: int = 16,
        metadata: dict | None = None,
        width: int = 320,
        height: int = 180,
    ):
        super().__init__(
            parameters={
                'mode': Mode.CHUNKED.value,
                'fps': 15.0,
                'sampling_period_sec': 1 / 15,
                'prediction_period_sec': action_horizon / 15,
                'history_offsets_sec': [-23 / 15, -16 / 15, -8 / 15, 0.0],
                'image_mappings': recipe.droid_cameras(),
                'jpeg_quality': 90,
            },
            metadata=metadata,
        )
        self._policy = policy
        self._max_in_flight = max_in_flight
        self._width, self._height = width, height
        self._active = False

    def _release(self) -> None:
        self._policy.reset()
        self._active = False

    def prepare_session(self, params: dict[str, Any]) -> DreamZeroSession:
        if self._active:
            raise RuntimeError('DreamZero supports one active session')
        mode = Mode(params['mode'])
        for name in ('fps', 'sampling_period_sec', 'prediction_period_sec'):
            value = params[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 1e-9:
                raise ValueError(f'{name} must be finite and positive')
        offsets = tuple(params['history_offsets_sec'])
        if (
            len(offsets) != 4
            or offsets[-1] != 0
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in offsets)
            or any(a >= b for a, b in pairwise(offsets))
        ):
            raise ValueError('history_offsets_sec must contain four increasing finite offsets ending in zero')
        mappings = params['image_mappings']
        if (
            not isinstance(mappings, dict)
            or set(mappings) != set(CAMERAS)
            or any(not isinstance(source, str) or not source for source in mappings.values())
        ):
            raise ValueError('image_mappings must map all three native cameras to observation fields')
        quality = params['jpeg_quality']
        check_jpeg_quality(quality, 'jpeg_quality')
        stack = recipe.client_stack(
            mode,
            fps=params['fps'],
            sampling_period_sec=params['sampling_period_sec'],
            prediction_period_sec=params['prediction_period_sec'],
            image_mappings=mappings,
            width=self._width,
            height=self._height,
            jpeg_quality=quality,
        )
        self._policy.reset()
        session = DreamZeroSession(
            self._policy,
            stack,
            max_in_flight=self._max_in_flight,
            sampling_period_sec=params['sampling_period_sec'],
            prediction_period_sec=params['prediction_period_sec'],
            history_offsets_sec=offsets,
            width=self._width,
            height=self._height,
            release=self._release,
        )
        self._active = True
        return session

    def close(self) -> None:
        self._policy.close()


def load_model(model_source: str, revision: str | None = None) -> DreamZeroModel:
    from .native import Policy  # noqa: PLC0415

    policy = Policy(model_source, revision)
    logger.info('Warming DreamZero before binding listeners')
    observation = {name: np.zeros((1, 180, 320, 3), dtype=np.uint8) for name in CAMERAS}
    observation.update({
        JOINTS: np.zeros((1, 7), np.float32),
        GRIP: np.zeros((1, 1), np.float32),
        PROMPT: 'Pick up the object.',
    })
    actions = policy.predict(observation)
    policy.reset()
    return DreamZeroModel(
        policy, action_horizon=len(actions[ACTION_JOINTS]), metadata={TYPE: 'dreamzero', CHECKPOINT_PATH: model_source}
    )


@cfn.config(model_source='GEAR-Dreams/DreamZero-DROID', revision=None, host='0.0.0.0', port=8000)
def serve(model_source: str, revision: str | None, host: str, port: int):
    server = ModelServer(partial(load_model, model_source, revision), auth_token=os.environ.get(AUTH_TOKEN_ENV))
    server.serve([WebsocketWire(ServedHostPort(host=host, port=port))])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfn.cli({'droid': serve.override(model_source=os.environ.get(MODEL_SOURCE_ENV, 'GEAR-Dreams/DreamZero-DROID'))})


if __name__ == '__main__':
    main()
