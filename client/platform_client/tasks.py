"""The catalogue task a rollout request names: `TaskRef('eight-spoons-into-grey-tote')`.

The platform owns the catalogue, so an id this client has never heard of still reaches the server;
what is refused here is a value that could never be a catalogue key.
"""

from __future__ import annotations

import re

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# The key's own shape, as the rollouts catalogue spells it (`rollouts_contract.tasks.is_task_id`):
# lowercase words joined by single hyphens, at most five. Spelled here too, since neither
# distribution imports the other.
MAX_ID_WORDS = 5
_TASK_ID = re.compile(rf'[a-z0-9]+(?:-[a-z0-9]+){{0,{MAX_ID_WORDS - 1}}}')
TASK_ID_HINT = f'lowercase words joined by single hyphens, at most {MAX_ID_WORDS}'


class TaskRef(str):
    """The id of a task the rollouts catalogue holds, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> TaskRef:
        if not _TASK_ID.fullmatch(value):
            raise ValueError(f'not a task id: {value!r} ({TASK_ID_HINT})')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
