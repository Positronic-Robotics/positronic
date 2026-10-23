"""ABC-DiT, the bimanual YAM policy of the ABC project, served over positronic's session protocol. NOTICE states its
licence."""

# The request and reply fields of ABC's policy server (`deploy/serve_policy.py`), which `server.py` runs.
STATE = 'state'
IMAGES = 'images'
PROMPT = 'prompt'
NOISE = 'noise'
ACTIONS = 'actions'
# The camera names that ABC's ``images`` field carries.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
