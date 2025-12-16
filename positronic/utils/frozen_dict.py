import collections.abc as cabc
from types import MappingProxyType
from typing import Any


class frozen_keys_dict(cabc.MutableMapping):
    """Dictionary wrapper that freezes the set of keys but allows updating values.

    Note: The Python stdlib has no built-in mapping that allows mutating
    values while preventing key additions/removals. `MappingProxyType` makes
    the entire mapping read-only, which isn't suitable here, so we provide a
    minimal wrapper to enforce "frozen keys, mutable values".

    - Setting a value for an existing key is allowed and updates the backing dict.
    - Adding a new key raises TypeError.
    - Deleting any key raises TypeError.
    """

    def __init__(self, backing: dict[str, Any]):
        self._backing = backing

    def __getitem__(self, key):
        return self._backing[key]

    def __setitem__(self, key, value):
        if key not in self._backing:
            raise TypeError('keys are frozen; cannot add new key')
        self._backing[key] = value

    def __delitem__(self, key):
        raise TypeError('keys are frozen; cannot delete keys')

    def __iter__(self):
        return iter(self._backing)

    def __len__(self):
        return len(self._backing)

    def __repr__(self) -> str:
        return f'frozen_keys_dict({self._backing!r})'


def frozen_view(original: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Create a frozen view of a dictionary.

    Returns a read-only view of the dictionary. The view reflects changes to the
    original dictionary, but does not allow modifications through the view itself.
    """
    return MappingProxyType(original)
