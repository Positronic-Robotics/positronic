"""A pi0.5 checkpoint that a third party fine-tuned on ABC's real YAM data, served over positronic's session
protocol. README.md states its licence situation."""

# The request and reply fields of the author's openpi server (`inference/openpi_server.py`), which `server.py` runs.
STATE = 'state'
IMAGES = 'images'
PROMPT = 'prompt'
ACTIONS = 'actions'
# The camera roles that the `images` field carries.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
