"""Metadata fields written by the model-server wrapper."""

HOST = 'host'
PORT = 'port'
# The socket path a server bound instead of a host and a port. One of the two pairs is present, never both.
UDS = 'uds'
# True when the server takes this session's images through a shared-memory ring beside ``UDS``.
FRAME_RING = 'frame_ring'
LOCAL_STACK = 'local_stack'
MODEL_SERVER_VERSION = 'model_server_version'
SESSION_PARAMS = 'session_params'
EFFECTIVE_PARAMS = 'effective_params'
