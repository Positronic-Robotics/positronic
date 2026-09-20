"""Explicit client/server policy configuration and the registry of deliverable components.

Processors and codecs compose the client stack. Server codecs wrap ordinary inference calls.
Model sources load callable models on the server; they cannot be delivered to clients.

TODO: Migrate the remaining vendor pipeline configs to explicit Pipeline arguments.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import and_, or_
from typing import Any, cast

from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, JointDeltaAction
from positronic.policy.base import PAR, SEQ, Obs, Policy, Processor
from positronic.policy.codec import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    RestrictImageSize,
)
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.observation import ObservationCodec
from positronic.policy.sequential import Sequential


class Model(ABC):
    """A loaded inference callable and the resources it owns."""

    @abstractmethod
    def __call__(self, obs: Obs) -> Any: ...

    def meta(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        """Release the model's resources after all calls have finished."""
        return None


class ModelSource(ABC):
    """Configuration that discovers and loads models; loaded resources belong to the returned model."""

    @abstractmethod
    def get_models(self) -> list[str]: ...

    def resolve(self, model_id: str | None) -> str:
        models = self.get_models()
        if model_id is None:
            return models[-1]
        if model_id not in models:
            raise ValueError(f'Unknown model {model_id!r}. Available: {models}')
        return model_id

    @abstractmethod
    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model: ...

    def meta(self, model_id: str) -> dict[str, Any]:
        return {}

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__


@dataclass
class Pipeline:
    """A model source, a client stack of processors and codecs, and an optional server codec."""

    source: ModelSource
    local: Policy
    codec: Codec | None = None
    compress_images: bool = False

    def __post_init__(self) -> None:
        if (
            self.codec is not None
            and roboarm_keys.EE_FRAME in self.codec.meta
            and roboarm_keys.EE_FRAME in self.local.meta()
        ):
            raise ValueError('Only one side of a pipeline may convert the end-effector frame')


WIRE_PROCESSORS = {processor.WIRE_NAME: processor for processor in (ChunkedSchedule, StopOnFault, TemporalStack)}
WIRE_CODECS = {
    codec.WIRE_NAME: codec
    for codec in (
        BinarizeGripTraining,
        BinarizeGripInference,
        FlipGrip,
        RestrictImageSize,
        ChangeEEFrame,
        ObservationCodec,
        AbsolutePositionAction,
        AbsoluteJointsAction,
        JointDeltaAction,
    )
}


def from_spec(node: dict[str, Any]) -> Processor | Codec:
    """Build registered processors or codecs from plain data. Unknown names and arguments raise."""
    if SEQ in node:
        parts = [from_spec(child) for child in node[SEQ]]
        if not parts:
            raise ValueError('A sequential spec must contain at least one component')
        if all(isinstance(part, Codec) for part in parts):
            return reduce(or_, cast(list[Codec], parts))
        return Sequential(parts[0], *parts[1:])
    if PAR in node:
        parts = [from_spec(child) for child in node[PAR]]
        if not parts or not all(isinstance(part, Codec) for part in parts):
            raise ValueError('Parallel specs require at least one codec')
        return reduce(and_, cast(list[Codec], parts))
    name = node.get('name')
    registered = WIRE_PROCESSORS | WIRE_CODECS
    if name not in registered:
        raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(registered)}')
    return registered[name](**node.get('args', {}))
