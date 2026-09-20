"""The keys of the server's own entries in the meta it hands a client."""

# The server's own entries in the ``META`` it hands over: where it serves, which checkpoint it resolved, and
# what the rig builds and obeys — the local stack spec, image compression, the positronic version it runs.
HOST = 'host'
PORT = 'port'
CHECKPOINT_ID = 'checkpoint_id'
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'
# Whether the server takes the frames of a temporal stack ahead of the observation, one per message.
STREAM_FRAMES = 'stream_frames'
