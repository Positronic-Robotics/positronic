"""Container-image references — the one type a caller and the platform pin a policy image with."""

from __future__ import annotations

import re

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# `<algorithm>:<hex>`, matched whole. The length is the registry's business — what is checked here
# is the shape a typo breaks: a missing half, or an encoding that is not hex.
_DIGEST = re.compile(r'[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*:[0-9a-fA-F]+')


class ImageRef(str):
    """A registry reference, `name[@digest]`, as a validated `str` — no serializer, no DB adapter.

    `name` is everything before the digest (repository plus any tag); `digest` is what pins the bytes.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> ImageRef:
        name, sep, digest = value.partition('@')
        if not name or any(c.isspace() for c in value) or (sep and not digest) or '@' in digest:
            raise ValueError(f'not an image reference: {value!r}')
        # A tag or a digest with nothing behind it is a typo, and one the registry only refuses at
        # submission time — where an unpullable image is a charged terminal submission.
        if sep and not _DIGEST.fullmatch(digest):
            raise ValueError(f'not a digest: {digest!r}')
        # A trailing colon is an empty tag. A registry port (`reg.io:5000/org/policy`) is not one:
        # what follows its colon is the rest of the path, so only an EMPTY remainder is the typo.
        if name.rpartition(':')[1] and not name.rpartition(':')[2]:
            raise ValueError(f'image reference has an empty tag: {value!r}')
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
