from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

T = TypeVar('T')
U = TypeVar('U')


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

    Implementation notes:
    - Lazy entries live in `_lazy_getters`, NOT in the underlying dict storage.
      Once computed, the value moves into dict storage and the getter stays but
      is never called again (guarded by `super().__contains__` check).
    - We override all dict read paths (__getitem__, get, __contains__, __iter__,
      __len__, keys, values, items) so that lazy keys are visible everywhere.
      Without this, consumers like `dict(lazy_dict)` or JSON serializers would
      silently drop uncomputed lazy keys.
    - __iter__ yields lazy keys WITHOUT triggering computation — this allows
      iteration/len checks to remain cheap. Only __getitem__/get/values/items
      trigger computation (when the value is actually needed).
    - copy() must bypass our overridden __iter__/__getitem__ to avoid triggering
      computation. It uses `dict.__iter__`/`dict.__getitem__` directly to read
      only the eagerly stored data, then passes unevaluated getters to the new
      LazyDict.

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

    def __iter__(self):
        # Yields lazy keys without triggering computation
        yield from super().__iter__()
        for k in self._lazy_getters:
            if not super().__contains__(k):
                yield k

    def __len__(self):
        return len(set(super().keys()) | set(self._lazy_getters.keys()))

    def keys(self):
        return set(super().keys()) | set(self._lazy_getters.keys())

    def values(self):
        # Triggers computation for all lazy keys
        return [self[k] for k in self]

    def items(self):
        # Triggers computation for all lazy keys
        return [(k, self[k]) for k in self]

    def copy(self):
        # Use dict internals directly to avoid triggering lazy computation.
        # dict.copy() and dict(self) would go through our __iter__/__getitem__
        # which would force evaluation of all lazy keys.
        eager_data = {k: dict.__getitem__(self, k) for k in dict.__iter__(self)}
        unevaluated = {k: v for k, v in self._lazy_getters.items() if k not in eager_data}
        return LazyDict(eager_data, unevaluated)


class LazySequence(Sequence[U]):
    """Lazy, indexable view that applies `fn` on element access.

    - Supports `len()` and integer indexing.
    - Slicing returns another lazy view without materializing elements.
    """

    def __init__(self, seq: Sequence[T], fn: Callable[[T], U]) -> None:
        self._seq = seq
        self._fn = fn

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, index: int | slice) -> U | LazySequence[U]:
        if isinstance(index, slice):
            return LazySequence(self._seq[index], self._fn)
        return self._fn(self._seq[int(index)])


def lazy_sequence(fn: Callable[[T], U]) -> Callable[[Sequence[T]], Sequence[U]]:
    """Decorator that wraps an elementwise transform into a lazy sequence transform."""
    return partial(LazySequence, fn=fn)
