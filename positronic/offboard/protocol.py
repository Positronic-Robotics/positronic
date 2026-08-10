"""The offboard session wire protocol: the field names and status values both ends spell.

- A session opens with status frames until ``READY``, then carries observations one way, results the other.
- `server.PolicyServer` writes these names and `client.InferenceSession` reads them, across two processes
  that share no code path and may run different versions of this package.
- The status values are the server's vocabulary, not a closed set: an unknown one is reported, not guessed.
"""

# Frame fields.
STATUS = 'status'
MESSAGE = 'message'
META = 'meta'
RESULT = 'result'
ERROR = 'error'

# Status values, in the order a session passes through them.
WAITING = 'waiting'  # queued behind another session holding the model slot or the backend
LOADING = 'loading'  # the checkpoint is being loaded; carries a progress MESSAGE
READY = 'ready'  # the model is loaded and a session is reset — the server can serve, and only now
STATUS_ERROR = 'error'  # the session failed to come up; the reason is in ERROR

# Not yet, keep waiting — a session in flight rather than one that failed.
PENDING_STATUSES = (WAITING, LOADING)
