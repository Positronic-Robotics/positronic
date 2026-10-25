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
    EndCommand,
    FlipGrip,
    Metadata,
    TransformImages,
)
from positronic.policy.codecs.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    CartesianDeltaAction,
    JointDeltaAction,
    JointPositionAction,
    SetControlMode,
    SplitActionChunk,
)
from positronic.policy.codecs.geometry import ConvertPose
from positronic.policy.codecs.observation import (
    PackFields,
    RenameObservationFields,
    SelectObservationFields,
    StateObservationCodec,
)
from positronic.policy.compatibility import (
    ObservationCodecV1,
    chunked_schedule_v2,
    pack_observation_fields_v1,
    restrict_image_size_v1,
    unpack_action_chunk_v1,
)
from positronic.policy.processors import ChunkedSchedule, EndOnField, PauseOnUnavailable, RTCSchedule, TemporalStack
from positronic.policy.sequential import Sequential
from positronic.utils.versions import Version, resolve_version

ComponentFactory = Callable[..., Processor | Codec | dict[str, Any]]


COMPONENTS: dict[str, dict[int, Version[ComponentFactory]]] = {
    component.WIRE_NAME: {component.WIRE_VERSION: Version(component.from_spec_args)}
    for component in (
        ChunkedSchedule,
        EncodeImages,
        EndCommand,
        EndOnField,
        RTCSchedule,
        PauseOnUnavailable,
        TemporalStack,
        BinarizeGripTraining,
        BinarizeGripInference,
        FlipGrip,
        Metadata,
        TransformImages,
        ChangeEEFrame,
        ConvertPose,
        ObservationCodecV1,
        StateObservationCodec,
        SelectObservationFields,
        PackFields,
        RenameObservationFields,
        AbsolutePositionAction,
        CartesianDeltaAction,
        AbsoluteJointsAction,
        JointDeltaAction,
        JointPositionAction,
        SetControlMode,
        SplitActionChunk,
    )
}


COMPONENTS.update({
    'pack_observation_fields': {1: Version(pack_observation_fields_v1)},
    'unpack_action_chunk': {1: Version(unpack_action_chunk_v1)},
    'restrict_image_size': {1: Version(restrict_image_size_v1)},
})
COMPONENTS[ChunkedSchedule.WIRE_NAME][2] = Version(chunked_schedule_v2)


def from_spec(node: dict[str, Any]) -> Processor | Codec:
    """Resolve versioned components and expand config translations before constructing their replacements.

    Missing versions mean v1; deprecations warn once per component version in this stack.
    """
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
        result = versions[version](**node.get(ARGS, {}))
        return build(result) if isinstance(result, dict) else result

    return build(node)
