"""The license rule: training data keeps a record only when each source that drove or that a step used has a license
that the caller allows.

The sources are the names in ``control.source`` and in ``control.uses``. The license of a source is at
``inference.<name>.license``, from the ``meta()`` of a local source, and at ``inference.<name>.server.license``, from
the server metadata of a remote source::

    rule = LicenseRule({'apache-2.0', 'internal'})
    if refused := rule.refused(record):
        ...  # drop the record; `refused` gives a refused license of each source that the rule refuses
"""

from collections.abc import Collection

from positronic.dataset.episode import Episode
from positronic.policy.keys import CONTROL_SOURCE, CONTROL_USES, LICENSE, NO_LICENSE, SERVER, SOURCES
from positronic.policy.record import decode_uses


def license_keys(name: str) -> tuple[str, str]:
    """The keys of the license of the source ``name`` in a record: from its ``meta()``, and from its server
    metadata."""
    prefix = f'{SOURCES}.{name}.'
    return prefix + LICENSE, f'{prefix}{SERVER}.{LICENSE}'


class LicenseRule:
    """The license rule, with the licenses that allow training in ``allowed``."""

    def __init__(self, allowed: Collection[str]):
        if isinstance(allowed, str):
            raise TypeError(
                f'The licenses that allow training are a collection of strings, such as [{allowed!r}], not the '
                f'string {allowed!r}'
            )
        if not allowed:
            raise ValueError('No license allows training, so the rule drops each record. Give at least one license')
        if NO_LICENSE in allowed:
            raise ValueError(f'{NO_LICENSE!r} is a missing license, and a missing license does not allow training')
        self._allowed = frozenset(allowed)

    def allows(self, license: str) -> bool:
        return license in self._allowed

    @staticmethod
    def _sources(record: Episode) -> set[str]:
        """The name of each source that drove or that a step used in ``record``."""
        sources = set(record[CONTROL_SOURCE].values()) if CONTROL_SOURCE in record else set()
        if CONTROL_USES in record:
            for uses in record[CONTROL_USES].values():
                sources |= decode_uses(uses)
        return sources

    @staticmethod
    def _licenses(record: Episode, name: str) -> list[str]:
        """Each license that ``record`` has for the source ``name``, or ``[NO_LICENSE]`` when it has none. A value that
        is not a string is ``NO_LICENSE``."""
        stated = [record[key] for key in license_keys(name) if key in record]
        return [value if isinstance(value, str) else NO_LICENSE for value in stated] or [NO_LICENSE]

    def refused(self, record: Episode) -> dict[str, str]:
        """A license of each source of ``record`` that the rule refuses, by name: the first of its licenses that is not
        in ``allowed``. A source with no license, or with no entry in the record, has ``NO_LICENSE``."""
        refused = {}
        for name in sorted(self._sources(record)):
            if not_allowed := [value for value in self._licenses(record, name) if not self.allows(value)]:
                refused[name] = not_allowed[0]
        return refused
