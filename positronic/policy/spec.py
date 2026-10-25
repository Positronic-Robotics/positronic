"""Registry and wire-spec decoding for client-side processors and codecs."""

from collections.abc import Callable
from functools import reduce
from operator import and_, or_
from typing import Any, cast

from positronic_model_server.spec import ARGS, LATEST, NAME, PAR, SEQ, VERSION

from positronic.policy.base import Processor
from positronic.policy.codecs import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    EncodeImages,
    FlipGrip,
    Metadata,
    RestrictImageSize,
)
from positronic.policy.codecs.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    CartesianDeltaAction,
    JointDeltaAction,
    JointPositionAction,
    SetControlMode,
    UnpackActionChunk,
)
from positronic.policy.codecs.geometry import ConvertPose
from positronic.policy.codecs.observation import ObservationCodec, PackObservationFields, RenameObservationFields
from positronic.policy.compatibility import ChunkedScheduleV2
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable, Prefetch, TemporalStack
from positronic.policy.sequential import Sequential
from positronic.utils.versions import Version, resolve_version

ComponentFactory = Callable[..., Processor | Codec]


COMPONENTS: dict[str, dict[int, Version[ComponentFactory]]] = {
    component.WIRE_NAME: {component.WIRE_VERSION: Version(component.from_spec_args)}
    for component in (
        ChunkedSchedule,
        Prefetch,
        EncodeImages,
        PauseOnUnavailable,
        TemporalStack,
        BinarizeGripTraining,
        BinarizeGripInference,
        FlipGrip,
        Metadata,
        RestrictImageSize,
        ChangeEEFrame,
        ConvertPose,
        ObservationCodec,
        PackObservationFields,
        RenameObservationFields,
        AbsolutePositionAction,
        CartesianDeltaAction,
        AbsoluteJointsAction,
        JointDeltaAction,
        JointPositionAction,
        SetControlMode,
        UnpackActionChunk,
    )
}


COMPONENTS[ChunkedSchedule.WIRE_NAME][ChunkedScheduleV2.WIRE_VERSION] = Version(ChunkedScheduleV2.from_spec_args)


def from_spec(node: dict[str, Any]) -> Processor | Codec:
    """Resolve latest or pinned versions. Missing versions mean v1; warn once per component version."""
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
        if version == LATEST:
            supported = [number for number, entry in COMPONENTS[name].items() if entry.implementation is not None]
            if not supported:
                raise ValueError(f'No supported versions of component {name!r}. Upgrade the client.')
            version = max(supported)
        versions = selected.setdefault(name, {})
        if type(version) is not int or version not in versions:
            factory = resolve_version(COMPONENTS[name], version, f'component {name!r}')
            versions[version] = factory
        args = dict(node.get(ARGS, {}))
        if name == Prefetch.WIRE_NAME:
            child = build(args[Prefetch.CHILD_ARG])
            if not isinstance(child, Processor):
                raise ValueError('Prefetch child must be a processor')
            args[Prefetch.CHILD_ARG] = child
        return versions[version](**args)

    return build(node)
