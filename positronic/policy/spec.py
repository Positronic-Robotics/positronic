"""Registry and wire-spec decoding for client-side processors and codecs."""

from functools import reduce
from operator import and_, or_
from typing import Any, cast

from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, JointDeltaAction
from positronic.policy.base import ARGS, NAME, PAR, SEQ, Processor
from positronic.policy.codec import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    Metadata,
    RestrictImageSize,
)
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.observation import ObservationCodec
from positronic.policy.sequential import Sequential

WIRE_PROCESSORS = {processor.WIRE_NAME: processor for processor in (ChunkedSchedule, StopOnFault, TemporalStack)}
WIRE_CODECS = {
    codec.WIRE_NAME: codec
    for codec in (
        BinarizeGripTraining,
        BinarizeGripInference,
        FlipGrip,
        Metadata,
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
    name = node.get(NAME)
    registered = WIRE_PROCESSORS | WIRE_CODECS
    if name not in registered:
        raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(registered)}')
    return registered[name](**node.get(ARGS, {}))
