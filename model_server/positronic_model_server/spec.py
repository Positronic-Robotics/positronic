"""Plain-data descriptions of versioned components and their composition."""

import json
from typing import Any

SEQ = 'seq'
PAR = 'par'
NAME = 'name'
ARGS = 'args'
VERSION = 'version'


def component(name: str, /, *, version: int = 1, **args: Any) -> dict[str, Any]:
    """Describe a component without importing it; arguments must survive a JSON round trip."""
    if not isinstance(name, str) or not name:
        raise ValueError('A component needs a nonempty name')
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
