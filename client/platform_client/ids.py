"""Platform id types: 64-bit ints in Python, bare lowercase hex on the wire.

`UserId`, `SubmissionId` and `RequestId` are distinct `Id64` subclasses, so a transposed argument
fails a typecheck rather than reaching the database; a service with ids of its own subclasses it
too. Range is `0 < value < 2**63`, and the wire form is never a JSON number — a full int64 does not
survive JavaScript's 2**53. `TransactionKey` / `ApiKey` are opaque tokens, not ids.
"""

from __future__ import annotations

import re
from typing import Any, NewType, Self

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# Exclusive upper bound: SQLite's INTEGER is signed 64-bit, so an id must fit a positive int64.
ID_LIMIT = 2**63

# The wire form, matched whole. `int(s, 16)` alone is too generous: it also takes a `0x` prefix, a
# sign and embedded underscores, so three spellings would name one id.
_HEX = re.compile(r'[0-9a-fA-F]+')


class Id64(int):
    """A platform id. Subclass it per resource; never use the base type in a signature."""

    __slots__ = ()

    def __new__(cls, value: int) -> Self:
        # bool is an int subclass, and `Id64(True)` is always a caller bug rather than the id 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'{cls.__name__} takes an int; use {cls.__name__}.parse() for a hex string')
        if not 0 < value < ID_LIMIT:
            raise ValueError(f'{cls.__name__} out of range: {value} (expected 0 < value < 2**63)')
        return super().__new__(cls, value)

    @classmethod
    def parse(cls, value: int | str) -> Self:
        """Accept the wire form (case-insensitive hex, no prefix) or a raw int."""
        if isinstance(value, str):
            if not _HEX.fullmatch(value):
                raise ValueError(f'{cls.__name__} must be bare hexadecimal: {value!r}')
            value = int(value, 16)
        return cls(value)

    def to_str(self) -> str:
        return format(int(self), 'x')

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return f'{type(self).__name__}.parse({self.to_str()!r})'

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            # JSON carries the hex string only, so a server that emitted a number is a hard error here
            # rather than a value silently truncated by whatever parsed it upstream.
            json_schema=core_schema.no_info_after_validator_function(cls.parse, core_schema.str_schema()),
            python_schema=core_schema.no_info_plain_validator_function(cls.parse),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.to_str, return_schema=core_schema.str_schema(), when_used='always'
            ),
        )


class UserId(Id64):
    """A registered user."""

    __slots__ = ()


class SubmissionId(Id64):
    """One user's run request, as the gateway records it."""

    __slots__ = ()


class RequestId(Id64):
    """One customer's rollout request, as the gateway records it."""

    __slots__ = ()


# A client-supplied dedup token, opaque to the platform: whatever the caller sends, echoed by nothing.
# Two submissions under one key are one transaction — the second returns the first, and is not charged.
TransactionKey = NewType('TransactionKey', str)

# The plaintext API key, returned by `users.register` exactly once and stored only as a hash.
ApiKey = NewType('ApiKey', str)
