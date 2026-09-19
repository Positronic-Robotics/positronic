"""Keys of the observation MolmoAct2 sessions take, which the codec writes and a warmup rebuilds."""

IMAGES = 'images'
STATE = 'state'
TASK = 'task'

# The (width, height) the model tiles every image to. A warm frame carries it, so its resize is a no-op.
IMAGE_SIZE = (378, 378)
