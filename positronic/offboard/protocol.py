"""The offboard session wire protocol: the field names and status values both ends spell.

A session opens with the server streaming status frames until it reports itself ready, after which
the connection carries observations one way and results the other. Those frame names are a contract
between two processes that share no code path — `server.PolicyServer` writes them, `client.
InferenceSession` reads them, and a rig and a vendor's server may be running different versions of
this package — so they are owned here and imported, rather than spelled at each end.

The status values are the server's vocabulary, not a closed type: a server this repository does not
ship may send one nothing here names, and the client answers that with `Unexpected server response`
rather than pretending to know what it meant.
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

# The statuses that mean "not yet, keep waiting" — a session in flight rather than one that failed.
PENDING_STATUSES = (WAITING, LOADING)
