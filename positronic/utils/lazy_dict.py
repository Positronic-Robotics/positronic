from __future__ import annotations

from collections.abc import Callable


class LazyDict(dict):
    """Dict that computes some keys lazily on first access.

    LazyDict behaves like a regular dict, but allows certain keys to be computed
    on-demand rather than upfront. This is useful when some values are expensive
    to compute and may not always be needed.

    How it works:
    - Initialize with eager `data` (computed immediately) and `lazy_getters`
      (functions that compute values when the key is first accessed)
    - Lazy keys appear in `keys()` and `in` checks, but aren't computed yet
    - On first access via `[]` or `.get()`, the getter is called and the result
      is cached in the dict (subsequent accesses use the cached value)
    - `copy()` preserves laziness: unevaluated keys remain lazy in the copy

    Example:
        >>> def expensive_computation():
        ...     print('Computing...')
        ...     return 42
        >>> d = LazyDict({'a': 1}, {'b': expensive_computation})
        >>> 'b' in d  # True (no computation yet)
        True
        >>> d['a']  # Regular access
        1
        >>> d['b']  # Triggers computation
        Computing...
        42
        >>> d['b']  # Cached, no recomputation
        42
    """

    def __init__(self, data: dict, lazy_getters: dict[str, Callable[[], object]]):
        super().__init__(data)
        self._lazy_getters = lazy_getters

    def _maybe_compute(self, key) -> None:
        """Compute and cache a lazy value if key is lazy and not yet computed."""
        if not super().__contains__(key) and key in self._lazy_getters:
            self[key] = self._lazy_getters[key]()

    def __getitem__(self, key):
        self._maybe_compute(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._maybe_compute(key)
        return super().get(key, default)

    def __contains__(self, key):
        return super().__contains__(key) or key in self._lazy_getters

    def keys(self):
        return set(super().keys()) | set(self._lazy_getters.keys())

    def copy(self):
        computed_keys = set(dict.keys(self))
        unevaluated = {k: v for k, v in self._lazy_getters.items() if k not in computed_keys}
        return LazyDict(dict(self), unevaluated)
