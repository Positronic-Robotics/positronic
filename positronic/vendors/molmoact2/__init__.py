"""Keys of the observation MolmoAct2 sessions take, which the codec writes and a warmup rebuilds."""

IMAGES = 'images'
STATE = 'state'
TASK = 'task'

# The (width, height) the model tiles every image to.
IMAGE_SIZE = (378, 378)

# The state one observation carries: every joint of the DROID arm, then the gripper.
NUM_JOINTS = 7
STATE_DIM = NUM_JOINTS + 1
