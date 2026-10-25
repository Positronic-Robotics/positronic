"""Session-local array prefixes addressed through ordinary inference inputs and results."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

import numpy as np

CACHE = 'cache'
REFERENCES = 'cached_values'
MISSING = 'missing_cached_values'


class SessionCache:
    """Hold the latest uploaded batch until another batch replaces it or the session ends."""

    def __init__(self) -> None:
        self._values: dict[str, np.ndarray] = {}

    async def call(self, obs: Mapping[str, Any], infer: Callable[[dict[str, Any]], Awaitable[Any]]) -> Any:
        if CACHE in obs:
            values = obs[CACHE]
            if not isinstance(values, Mapping) or any(
                not isinstance(v, np.ndarray) or v.ndim == 0 or len(v) == 0 for v in values.values()
            ):
                raise ValueError('A cache request contains named, nonempty array prefixes')
            references = {key: uuid4().hex for key in values}
            self._values = {references[key]: value for key, value in values.items()}
            return references

        request = dict(obs)
        references = request.pop(REFERENCES, {})
        if any(ref not in self._values for ref in references.values()):
            return {MISSING: True}
        for key, ref in references.items():
            request[key] = np.concatenate((self._values[ref], request[key]), axis=0)
        return await infer(request)
