"""The keys of the server's own entries in what it tells a client about itself."""

# The server's own entries in the ``META`` it hands over: where it serves, which checkpoint it resolved, and
# what the rig builds and obeys — the local stack spec, image compression, the positronic version it runs.
# ``CHECKPOINT_ID`` and ``POSITRONIC_VERSION`` name the same two facts in the readiness record, which a
# caller reads without opening a session.
HOST = 'host'
PORT = 'port'
CHECKPOINT_ID = 'checkpoint_id'
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'
