"""Shared policy metadata and observation field names."""

# What a policy reports about itself through its ``meta``; a remote policy nests the server's meta under
# ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy at the top
# level and the vendor under ``SERVER``, so a reader composes a prefix with a field: f'{SERVER_META}.{TYPE}'.
# The block under ``SERVER`` is the server's handshake metadata as sent. A ``prompt`` in it is not the task.
TYPE = 'type'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
ACTION_FPS = 'action_fps'
ACTION_HORIZON_SEC = 'action_horizon_sec'
SERVER = 'server'

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'

OBS_TIME_NS = 'obs_time_ns'
