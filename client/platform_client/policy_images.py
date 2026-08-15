"""Policy images — the container image a submission's policy is run from."""

from __future__ import annotations

import re

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# The reference grammar a registry parses: an optional host and port, a lowercase repository path,
# an optional tag, an optional digest. A typo it admits is an unpullable image — terminal, and CHARGED.
_HOST = r'[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?'
_DOMAIN = rf'{_HOST}(?:\.{_HOST})*(?::[0-9]+)?'
_PATH = r'[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*'
_TAG = r'[A-Za-z0-9_][A-Za-z0-9._-]{0,127}'
# `<algorithm>:<hex>`. The length is the registry's business — what is checked is the shape a typo
# breaks: a missing half, or an encoding that is not hex.
_DIGEST = r'[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*:[0-9a-fA-F]+'
_REFERENCE = re.compile(rf'(?:{_DOMAIN}/)?{_PATH}(?:/{_PATH})*(?::{_TAG})?(?:@{_DIGEST})?')


class PolicyImage(str):
    """A registry reference, `name[@digest]`, as a validated `str` — no serializer, no DB adapter.

    `name` is everything before the digest (repository plus any tag); `digest` is what pins the bytes.

    The platform pulls this reference as given, when the run starts, and evaluates what it finds.
    The submitter keeps it pullable, and keeps it holding what they want evaluated, until the run
    finishes; a tag resolves at pull time, so pin a digest for a fixed image.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> PolicyImage:
        if not _REFERENCE.fullmatch(value):
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

    def pinned(self, digest: str) -> PolicyImage:
        """This reference bound to `digest`, REPLACING any digest it already carries — appending
        instead would turn an already-pinned `repo@sha256:…` into an unpullable double digest."""
        return PolicyImage(f'{self.name}@{digest}')
