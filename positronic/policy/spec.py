"""Registry and wire-spec decoding for client-side processors and codecs."""

from collections.abc import Callable
from functools import reduce
from operator import and_, or_
from typing import Any, cast

from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, JointDeltaAction
from positronic.policy.base import ARGS, NAME, PAR, SEQ, VERSION, Processor
from positronic.policy.codec import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    Metadata,
    RestrictImageSize,
)
from positronic.policy.compatibility import (
    ActionHorizonV1,
    ActionTimestampV1,
    ChunkedScheduleV1,
    StackV1,
    StopOnFaultV1,
    TemporalStackV1,
    _LayerV1,
)
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.observation import ObservationCodec
from positronic.policy.sequential import Sequential
from positronic.utils.versions import Version, resolve_version

ComponentFactory = Callable[..., Processor | Codec]

COMPONENTS: dict[str, dict[int, Version[ComponentFactory]]] = {
    component.WIRE_NAME: {component.WIRE_VERSION: Version(component)}
    for component in (
        ChunkedSchedule,
        PauseOnUnavailable,
        TemporalStack,
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
for component in (ChunkedScheduleV1, StopOnFaultV1, TemporalStackV1, ActionTimestampV1, ActionHorizonV1):
    COMPONENTS.setdefault(component.WIRE_NAME, {})[component.WIRE_VERSION] = Version(component)


def from_spec(node: dict[str, Any]) -> Processor | Codec:
    """Build an exact versioned stack. Unversioned components mean v1; warn once per component version."""
    selected: dict[str, dict[int, ComponentFactory]] = {}

    def build(node: dict[str, Any]) -> Processor | Codec:
        if SEQ in node:
            parts = [build(child) for child in node[SEQ]]
            if not parts:
                raise ValueError('A sequential spec must contain at least one component')
            if all(isinstance(part, Codec) for part in parts):
                return reduce(or_, cast(list[Codec], parts))
            composition = StackV1 if any(isinstance(part, (_LayerV1, StackV1)) for part in parts) else Sequential
            return composition(parts[0], *parts[1:])
        if PAR in node:
            parts = [build(child) for child in node[PAR]]
            if not parts or not all(isinstance(part, Codec) for part in parts):
                raise ValueError('Parallel specs require at least one codec')
            return reduce(and_, cast(list[Codec], parts))
        name, version = node.get(NAME), node.get(VERSION, 1)
        if not isinstance(name, str) or name not in COMPONENTS:
            raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(COMPONENTS)}')
        versions = selected.setdefault(name, {})
        if type(version) is not int or version not in versions:
            factory = resolve_version(COMPONENTS[name], version, f'component {name!r}')
            versions[version] = factory
        return versions[version](**node.get(ARGS, {}))

    stack = build(node)
    legacy_timing = any(
        codec.WIRE_VERSION in selected.get(codec.WIRE_NAME, {}) for codec in (ActionTimestampV1, ActionHorizonV1)
    )
    if legacy_timing and isinstance(stack, Processor) and not isinstance(stack, (_LayerV1, StackV1)):
        raise ValueError('V1 timing codecs cannot be mixed with Step processors; configure ChunkedSchedule timing')
    return stack
