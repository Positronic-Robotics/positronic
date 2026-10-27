"""ABC-DiT, the bimanual YAM policy of the ABC project, served over positronic's session protocol. NOTICE states its
licence."""

# The request and reply fields of ABC's policy server (`deploy/serve_policy.py`), which `server.py` runs.
STATE = 'state'
IMAGES = 'images'
PROMPT = 'prompt'
NOISE = 'noise'
ACTIONS = 'actions'
# The request fields of ABC's real-time chunking: the actions the robot plays while the policy computes, and how many
# of them there are. The new chunk starts with those actions.
ACTION_PREFIX = 'action_prefix'
PREFIX_LENGTH = 'prefix_length'
# A request field the codec sets for `server.py` only: how many actions of each chunk the client plays. The server
# removes it before ABC's server gets the request.
EXECUTED_ACTIONS = 'executed_actions'
# The camera names that ABC's ``images`` field carries.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
