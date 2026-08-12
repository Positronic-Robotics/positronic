"""Container-image references — the one type a caller and the platform pin a policy image with."""

from __future__ import annotations

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ImageRef(str):
    """A registry reference, `name[@digest]`, as a validated `str` — no serializer, no DB adapter.

    `name` is everything before the digest (repository plus any tag); `digest` is what pins the bytes.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> ImageRef:
        name, sep, digest = value.partition('@')
        if not name or any(c.isspace() for c in value) or (sep and not digest) or '@' in digest:
            raise ValueError(f'not an image reference: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())

    @property
    def name(self) -> str:
        return self.partition('@')[0]

    @property
    def digest(self) -> str | None:
        return self.partition('@')[2] or None

    def pinned(self, digest: str) -> ImageRef:
        """This reference bound to `digest`, REPLACING any digest it already carries — appending
        instead would turn an already-pinned `repo@sha256:…` into an unpullable double digest."""
        return ImageRef(f'{self.name}@{digest}')
