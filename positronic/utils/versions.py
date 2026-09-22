"""Exact version selection and dated deprecation notices for published contracts."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class Deprecation:
    announced_on: date
    remove_after: date
    replacement: str

    def __post_init__(self) -> None:
        if self.remove_after <= self.announced_on:
            raise ValueError('Removal must follow the deprecation announcement')
        if not self.replacement:
            raise ValueError('A deprecation must explain how to migrate')


@dataclass(frozen=True)
class Version(Generic[T]):
    """An implementation, or a removed version retained for its migration instructions.

    Dates announce the earliest removal; they never disable an installed implementation.
    """

    implementation: T | None
    deprecation: Deprecation | None = None
    removed_on: date | None = None

    def __post_init__(self) -> None:
        if self.implementation is None:
            if self.deprecation is None or self.removed_on is None:
                raise ValueError('A removed version must retain its deprecation notice and removal date')
            if self.removed_on < self.deprecation.remove_after:
                raise ValueError('A version cannot be removed before its announced deadline')
        elif self.removed_on is not None:
            raise ValueError('A supported implementation cannot have a removal date')

    def resolve(self, label: str) -> T:
        if self.deprecation is not None:
            notice = self.deprecation
            if self.implementation is None:
                raise ValueError(f'{label} has been removed. {notice.replacement}')
            warnings.warn(
                f'{label} is deprecated since {notice.announced_on}; support may be removed in a client release '
                f'on or after {notice.remove_after}. {notice.replacement}',
                FutureWarning,
                stacklevel=3,
            )
        assert self.implementation is not None
        return self.implementation


def resolve_version(versions: Mapping[int, Version[T]], version: object, label: str) -> T:
    """Resolve exactly the requested version; no upgrade or fallback is implicit."""
    if type(version) is not int or version < 1:
        raise ValueError(f'{label} version must be a positive integer, got {version!r}')
    if version not in versions:
        supported = [number for number, entry in versions.items() if entry.implementation is not None]
        raise ValueError(
            f'Unsupported {label} version {version}; supported versions: {sorted(supported)}. Upgrade the client.'
        )
    return versions[version].resolve(f'{label} v{version}')
