"""Eval references — the one name a caller picks a run by.

An eval names a task suite AND the embodiment that runs it: `franka.spoons` is the wooden-spoon
task on a real Franka, and nothing else can run it. So the caller chooses one name and the platform
resolves everything behind it; there is no second axis to get wrong.

The platform owns the set of names. This type refuses only what could never be one, so a name this
client has never heard of still reaches the server, which answers with the ones it offers.
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
