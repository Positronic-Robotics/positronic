"""Transform values inside mappings, lists and tuples without mutating their containers."""

from collections.abc import Callable, Mapping
from typing import Any


def map_leaves(value: Any, transform: Callable[[Any], Any]) -> Any:
    """Rebuild containers and transform each leaf once; callback results are not traversed.

    Mappings become dictionaries. Lists and tuples retain their type. Every other value,
    including an array, is a leaf. Empty containers remain present.
    """
    if isinstance(value, Mapping):
        return {key: map_leaves(item, transform) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(map_leaves(item, transform) for item in value)
    return transform(value)


def update_at(value: Any, path: tuple[str | int, ...], transform: Callable[[Any], Any]) -> Any:
    """Transform the selected value, copying only containers along its path.

    An empty path selects the whole value, including a container. Keys are literal;
    lists and tuples take integer indices. Missing keys and indices raise.
    """
    if not path:
        return transform(value)
    key, *rest = path
    if isinstance(value, Mapping):
        return {**value, key: update_at(value[key], tuple(rest), transform)}
    if isinstance(value, list | tuple) and type(key) is int:
        result = list(value)
        result[key] = update_at(value[key], tuple(rest), transform)
        return tuple(result) if isinstance(value, tuple) else result
    raise ValueError(f'Path {path!r} does not match the value')
