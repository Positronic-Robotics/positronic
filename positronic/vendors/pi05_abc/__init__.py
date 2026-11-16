"""pi0.5 checkpoints fine-tuned on ABC's real YAM data, served over positronic's session protocol. README.md states
the licence situation."""

# The request and reply fields of the author's openpi server (`yam_abc_reproduce/deploy/servers/openpi_server.py` in the
# fork), which `server.py` runs.
STATE = 'state'
IMAGES = 'images'
PROMPT = 'prompt'
ACTIONS = 'actions'
# The camera roles that the `images` field carries.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
