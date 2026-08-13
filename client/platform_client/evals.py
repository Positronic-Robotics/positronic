"""The one name a caller picks a run by: `EvalRef('franka.spoons')`.

A name is a task suite AND the embodiment that runs it, so there is no second axis to get wrong.
The platform owns the set, so a name this client has never heard of still reaches the server.
"""

from __future__ import annotations

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class EvalRef(str):
    """The name of an eval the platform offers, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> EvalRef:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not an eval name: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
