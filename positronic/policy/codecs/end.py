from typing import Any

from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic.policy.keys import END

from .base import Codec


class EndCommand(Codec):
    """Decode a model's per-element end decision in ``field`` into the reserved command ``END``.

    Compose it in parallel with the action codec, ``action & EndCommand()``, so each decoded element carries
    ``END`` beside its other commands. ``EndOnField`` in the client stack ends the run when that element plays.
    """

    WIRE_NAME = 'end_command'

    def __init__(self, field: str = END):
        self._field = field

    def _decode_single(self, data: dict) -> dict:
        return {END: data[self._field]} if self._field in data else {}

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'field': self._field}}
