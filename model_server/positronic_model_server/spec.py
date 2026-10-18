"""Plain-data component descriptions and session parameter utilities."""

import json
from collections.abc import Sequence
from typing import Any

SEQ = 'seq'
PAR = 'par'
NAME = 'name'
ARGS = 'args'
VERSION = 'version'

CHUNKED_SCHEDULE = 'chunked_schedule'
STOP_ON_FAULT = 'stop_on_fault'
TEMPORAL_STACK = 'temporal_stack'

CURRENT_COMPONENT_VERSIONS = {CHUNKED_SCHEDULE: 3, STOP_ON_FAULT: 2, TEMPORAL_STACK: 2}


def component(name: str, /, *, version: int | None = None, **args: Any) -> dict[str, Any]:
    """Pin an omitted version to the shared current version, defaulting to v1, without importing the component."""
    if not isinstance(name, str) or not name:
        raise ValueError('A component needs a nonempty name')
    if version is None:
        version = CURRENT_COMPONENT_VERSIONS.get(name, 1)
    if type(version) is not int or version < 1:
        raise ValueError('A component version must be a positive integer')
    # JSON normalizes tuples to lists and rejects Python objects and nonfinite numbers.
    return {NAME: name, VERSION: version, ARGS: json.loads(json.dumps(args, allow_nan=False))}


def sequence(*parts: dict[str, Any]) -> dict[str, Any]:
    """Compose at least one processor or codec in execution order."""
    if not parts:
        raise ValueError('A sequential spec must contain at least one component')
    return {SEQ: list(parts)}


def parallel(*parts: dict[str, Any]) -> dict[str, Any]:
    """Describe codecs applied in parallel; the client checks that each component is a codec."""
    if not parts:
        raise ValueError('Parallel specs require at least one codec')
    return {PAR: list(parts)}


def validate(node: dict[str, Any]) -> None:
    """Check description structure without resolving client components."""
    if not isinstance(node, dict):
        raise ValueError('A component description must be a mapping')
    compositions = {SEQ, PAR} & node.keys()
    if compositions:
        if len(node) != 1:
            raise ValueError('A composition must contain only seq or par')
        parts = node[next(iter(compositions))]
        if not isinstance(parts, list) or not parts:
            raise ValueError('A composition needs a nonempty list of components')
        for part in parts:
            validate(part)
    else:
        if node.keys() - {NAME, VERSION, ARGS} or not isinstance(node.get(ARGS, {}), dict):
            raise ValueError('A component contains invalid fields or arguments')
        name = node.get(NAME)
        if not isinstance(name, str):
            raise ValueError('A component needs a name')
        version = node.get(VERSION, 1)
        if type(version) is not int or version < 1:
            raise ValueError('A component version must be a positive integer')
        component(name, version=version, **node.get(ARGS, {}))


def resolve_params(defaults: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    """Apply declared session overrides, returning independent JSON-compatible values."""
    unknown = requested.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown session parameters: {sorted(unknown)}')
    return json.loads(json.dumps({**defaults, **requested}, allow_nan=False))


def parse_params(items: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Decode query values as JSON or plain strings, rejecting duplicate parameter names."""
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError(f'Duplicate session parameter: {name}')
        try:
            result[name] = json.loads(value)
        except json.JSONDecodeError:
            result[name] = value
    return result
