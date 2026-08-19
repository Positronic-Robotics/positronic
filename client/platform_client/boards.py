"""The slug a ranking board is read by: `BoardRef('nebius-2026/robolab.smoke')`.

The platform owns the set, so a board this client has never heard of still reaches the server.
"""

from __future__ import annotations

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class BoardRef(str):
    """The slug of a board the platform publishes, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> BoardRef:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not a board slug: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
