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
            return Sequential(parts[0], *parts[1:])
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

    return build(node)
