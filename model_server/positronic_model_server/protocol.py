"""Shared inference message fields, versions, authentication and timing."""

from enum import IntEnum, StrEnum

AUTH_HEADER = 'Authorization'


def bearer(token: str) -> str:
    """The authorization header value for a bearer token."""
    return f'Bearer {token}'


# The top-level keys of every server-to-client message: ``STATUS`` until the server reports itself ready
# and hands over its ``META``, then one ``RESULT`` or ``ERROR`` per inference.
STATUS = 'status'
MESSAGE = 'message'
META = 'meta'
RESULT = 'result'
ERROR = 'error'
# The ready handshake issues an ID. Requests carry it beside the observation, and an end request
# is acknowledged with the same ID after the model releases the session's state.
SESSION_ID = 'session_id'
OBSERVATION = 'observation'
END_SESSION = 'end_session'
PROTOCOL_VERSION = 'protocol_version'


class ProtocolVersion(IntEnum):
    V1 = 1
    V2 = 2


# What the server spent on one inference, beside the ``RESULT`` it answers with: durations in
# milliseconds on the server's own clock. A server that sends none leaves the round trip undivided.
TIMING = 'timing'

# The phases ``TIMING`` reports. `SERVED` brackets the others: it opens on the observation
# arriving and closes before the answer is encoded.
TIMING_SERVED = 'served_ms'
TIMING_DECODE = 'decode_ms'
TIMING_INFER = 'infer_ms'
# Time the observation waited for the inference slot, inside `SERVED`.
TIMING_QUEUED = 'queued_ms'


def timing_key(name: str) -> str:
    return f'{name}_ms'


# The loaded model's call, excluding codec conversions.
MODEL_CALL = 'model'
# Time the model's own call took, inside `INFER`.
TIMING_MODEL = timing_key(MODEL_CALL)


class ServerStatus(StrEnum):
    READY = 'ready'
    WAITING = 'waiting'
    LOADING = 'loading'
    ERROR = 'error'
