"""The keys of the server's own entries in what it tells a client about itself."""

HOST = 'host'
PORT = 'port'
# The socket path a server bound instead of a host and a port. One of the two pairs is present, never both.
UDS = 'uds'
CHECKPOINT_ID = 'checkpoint_id'
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'
