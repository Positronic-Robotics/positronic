"""Support legacy processor specs and protocol v1 server stacks.

The offboard README states the translation and what the client refuses.
"""

import logging
import time
from collections.abc import Callable, Generator, Mapping
from typing import Any, overload

from positronic_model_server.spec import ARGS, NAME, SEQ, VERSION, component, sequence

from positronic import keys as obs_keys
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get
from positronic.dataset.transforms.image import Resample, ResizeMode
from positronic.policy import keys as policy_keys
from positronic.policy import training
from positronic.policy.base import Obs, Policy, PolicyRun, Processor, ProcessorRun, Runtime
from positronic.policy.codecs import (
    TASK_FIELD,
    Codec,
    PackFields,
    SplitActionChunk,
    StateObservationCodec,
    TransformImages,
)
from positronic.policy.codecs.image import image_observation
from positronic.policy.keys import ACTION_FPS
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.sequential import Sequential

TIMESTAMP = 'timestamp'
WALL_TIME_NS = 'wall_time_ns'
ACTION_TIMESTAMP = 'action_timestamp'
ACTION_HORIZON = 'action_horizon'
V1_FPS_ARG = 'fps'
V1_HORIZON_SEC_ARG = 'horizon_sec'
V1_LAYERS_UPGRADED_IN_PLACE = (PauseOnUnavailable.WIRE_NAME, TemporalStack.WIRE_NAME)
V1_SERVER_DEFAULT_ACTION_FPS = 15.0

logger = logging.getLogger(__name__)


class WithTraining(Codec):
    """Expose an independent training transform through the legacy codec interface.

    **TODO: TEMPORARY ADAPTER — MUST BE REMOVED.**
    Migrate remaining vendor presets and training consumers to independent recipes, following GR00T.
    Then delete this adapter together with Codec.training_encoder, its composition and apply_codec.
    """

    def __init__(self, inference: Codec, training: EpisodeTransform):
        self._inference = inference
        self._training = training

    def encode(self, data):
        return self._inference.encode(data)

    def decode(self, data):
        return self._inference.decode(data)

    @property
    def meta(self):
        return self._inference.meta

    @property
    def training_encoder(self):
        return self._training

    def to_spec(self):
        return self._inference.to_spec()


def pack_observation_fields_v1(layout: dict[str, Any], *, unsqueeze_dims: int = 0) -> dict[str, Any]:
    """Translate literal input field names into source paths for input-only packing."""

    def paths(group: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for name, source in group.items():
            if not isinstance(name, str) or not isinstance(source, (str, dict)):
                raise ValueError('Layout entries must have string keys and contain input key names or dictionaries')
            result[name] = paths(source) if isinstance(source, dict) else [source]
        return result

    return component(PackFields.WIRE_NAME, version=1, encode_layout=paths(layout), unsqueeze_dims=unsqueeze_dims)


def unpack_action_chunk_v1(fields: dict[str, list[str | int]], *, squeeze_dims: int = 0) -> dict[str, Any]:
    """Translate prediction selection and chunk splitting into independent operations."""
    if not fields or any(
        not isinstance(name, str) or not isinstance(path, (list, tuple)) for name, path in fields.items()
    ):
        raise ValueError('Prediction fields require a non-empty mapping of source paths')
    return sequence(
        component(SplitActionChunk.WIRE_NAME, version=1),
        component(PackFields.WIRE_NAME, version=1, decode_layout=fields, squeeze_dims=squeeze_dims),
    )


def restrict_image_size_v1(
    width: int = 640, height: int = 640, resample: str = Resample.BILINEAR.value
) -> dict[str, Any]:
    """Translate shrink-only image bounds, including their published defaults."""
    return component(
        TransformImages.WIRE_NAME, version=1, width=width, height=height, mode=ResizeMode.BOUND.value, resample=resample
    )


class ObservationCodecV1(Codec):
    """Decode the combined state/image observation schema published as version 1."""

    WIRE_NAME = 'observation_codec'

    def __init__(
        self,
        state: dict[str, dict[str, int]],
        images: dict[str, tuple[str, tuple[int, int]]],
        task_field: str = TASK_FIELD,
    ):
        self._state = StateObservationCodec(state, task_field)
        self._image_configs = images
        self._images = image_observation(images)
        self._state_names = tuple(state)
        self._task_field = task_field

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        # V1 inference gives state precedence over images, and images over the prompt.
        obs = {self._task_field: data[obs_keys.TASK]} if obs_keys.TASK in data else {}
        obs.update(self._images.encode(data))
        state = self._state.encode(data)
        obs.update({name: state[name] for name in self._state_names})
        return obs

    @property
    def training_encoder(self):
        # V1 training gives the task column precedence over images, and images over state.
        return (
            Derive(meta=None, **{TASK_FIELD: Get(obs_keys.TASK, '')})
            & training.image_observation(self._image_configs)
            & self._state.training_encoder
        )

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                **self._state.to_spec()[ARGS],
                'images': {name: [source, list(size)] for name, (source, size) in self._image_configs.items()},
            },
        }


def chunked_schedule_v2(
    fps: float,
    horizon_sec: float | None = None,
    record_stats: bool = True,
    inference_lead: float | dict[str, Any] = 0.0,
) -> dict[str, Any]:
    """Translate the scheduler that keeps its pending chunk after an execution error."""
    return component(
        ChunkedSchedule.WIRE_NAME,
        version=3,
        fps=fps,
        horizon_sec=horizon_sec,
        record_stats=record_stats,
        inference_lead=inference_lead,
        replan_on_error=False,
    )


class StampObservationTimes(Policy):
    """Add the control clock's ``obs_time_ns`` and the wall clock's ``wall_time_ns``, which a v1 server reads."""

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            stamped = {**obs, policy_keys.OBS_TIME_NS: runtime.time_ns, WALL_TIME_NS: time.time_ns()}
            try:
                step = inner.send(stamped)
            except StopIteration as end:
                return end.value
            obs = yield step


class ChunkFromV1Answer(Codec):
    """Turn a v1 answer into a chunk: ``None`` becomes no rows, a single row one row, and timestamps and the end row go.

    It wraps the call itself, because ``Codec.wrap`` does not decode ``None``.
    """

    @overload
    def wrap(self, function: Callable[[dict], Any]) -> Callable[[Obs], Any]: ...

    @overload
    def wrap(self, function: ProcessorRun[Obs, Any]) -> ProcessorRun[Obs, Any]: ...

    def wrap(
        self, function: Callable[[dict], Any] | ProcessorRun[Obs, Any]
    ) -> Callable[[Obs], Any] | ProcessorRun[Obs, Any]:
        if isinstance(function, Generator):
            return super().wrap(function)
        infer = function

        def answer_or_empty_chunk(obs: dict) -> Any:
            answer = infer(obs)
            return [] if answer is None else answer

        return super().wrap(answer_or_empty_chunk)

    def encode(self, data: dict) -> dict:
        return data

    def decode(self, data: Any) -> list[dict[str, Any]]:
        rows = [data] if isinstance(data, Mapping) else data
        return [{key: row[key] for key in row if key != TIMESTAMP} for row in rows if set(row) != {TIMESTAMP}]


def _unnest_seq(node: dict[str, Any]) -> list[dict[str, Any]]:
    if SEQ in node:
        return [part for child in node[SEQ] for part in _unnest_seq(child)]
    return [node]


def _is_v1(part: dict[str, Any], name: str) -> bool:
    return part.get(NAME) == name and part.get(VERSION, 1) == 1


def from_v1_spec(
    node: dict[str, Any], server_meta: Mapping[str, Any], build_stack: Callable[[dict[str, Any]], Processor | Codec]
) -> Processor[Obs, Any]:
    """Build the stack a protocol v1 server declares, from its spec and its handshake metadata."""
    timing: dict[str, float] = {}
    parts: list[dict[str, Any]] = []
    for part in _unnest_seq(node):
        if _is_v1(part, ACTION_TIMESTAMP):
            timing[ChunkedSchedule.FPS_ARG] = part[ARGS][V1_FPS_ARG]
        elif _is_v1(part, ACTION_HORIZON):
            timing[ChunkedSchedule.HORIZON_SEC_ARG] = part[ARGS][V1_HORIZON_SEC_ARG]
        elif part.get(VERSION, 1) == 1 and part.get(NAME) in V1_LAYERS_UPGRADED_IN_PLACE:
            parts.append({**part, VERSION: 2})
        else:
            parts.append(part)
    if ChunkedSchedule.FPS_ARG not in timing:
        if ACTION_FPS not in server_meta:
            logger.warning(
                'The v1 server declares no action_timestamp and sends no action_fps; the client assumes %s '
                'actions per second. Rebuild the server on current positronic so that it declares its rate.',
                V1_SERVER_DEFAULT_ACTION_FPS,
            )
        timing[ChunkedSchedule.FPS_ARG] = server_meta.get(ACTION_FPS, V1_SERVER_DEFAULT_ACTION_FPS)
    if ChunkedSchedule.HORIZON_SEC_ARG not in timing and server_meta.get(policy_keys.ACTION_HORIZON_SEC) is not None:
        timing[ChunkedSchedule.HORIZON_SEC_ARG] = server_meta[policy_keys.ACTION_HORIZON_SEC]

    schedule = {NAME: ChunkedSchedule.WIRE_NAME, VERSION: 2, ARGS: timing}
    positions = [i for i, part in enumerate(parts) if _is_v1(part, ChunkedSchedule.WIRE_NAME)]
    if positions:
        parts[positions[0]] = schedule
    else:
        layers = [i for i, part in enumerate(parts) if part.get(NAME) in V1_LAYERS_UPGRADED_IN_PLACE]
        parts.insert(layers[-1] + 1 if layers else 0, schedule)
    return Sequential(StampObservationTimes(), build_stack({SEQ: parts}), ChunkFromV1Answer())
