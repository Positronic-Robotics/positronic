"""pi0.5 checkpoints fine-tuned on ABC's real YAM data, served over positronic's session protocol. README.md states
the licence situation."""

# The request and reply fields of the author's openpi server (`yam_abc_reproduce/deploy/servers/openpi_server.py` in the
# fork), which `server.py` runs.
STATE = 'state'
IMAGES = 'images'
PROMPT = 'prompt'
# The last command that the arms got, in the layout of `state`. Only a checkpoint with the last command as its state
# input reads it.
LAST_COMMAND = 'last_command'
ACTIONS = 'actions'
# The camera roles that the `images` field carries.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
