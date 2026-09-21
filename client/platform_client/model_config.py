"""The pydantic config every model built from input carries: it declares what it takes, and its
errors do not repeat what it was given.

A model-level validator is handed the raw input whole, before any field is coerced, so a
`SecretStr` field masks nothing in the `ValidationError` that follows. `hide_input_in_errors`
covers the renderings the model controls — `str`, `repr`, and every traceback and log line built
from one — while `errors()` and `json()` take `include_input` from the caller instead.
"""

from __future__ import annotations

from pydantic import ConfigDict

INPUT_MODEL_CONFIG = ConfigDict(extra='forbid', hide_input_in_errors=True)
