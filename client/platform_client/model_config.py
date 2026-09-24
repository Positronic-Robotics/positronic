"""The config for every model built from caller input: an unknown field is refused, and a
`ValidationError` does not echo the input.

A `SecretStr` field masks nothing in a `ValidationError`, because pydantic validates a model
over its raw input. The hiding reaches `str` and `repr`, and `errors()` and `json()` take
`include_input` from the caller.
"""

from __future__ import annotations

from pydantic import ConfigDict

INPUT_MODEL_CONFIG = ConfigDict(extra='forbid', hide_input_in_errors=True)
