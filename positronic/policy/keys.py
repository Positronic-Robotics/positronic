"""Shared policy metadata and observation field names."""

# ``TYPE`` names the policy, or the vendor under ``SERVER``. ``SERVER`` holds a wire server's handshake metadata as
# sent, so a ``prompt`` there is not the task.
TYPE = 'type'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
ACTION_FPS = 'action_fps'
ACTION_HORIZON_SEC = 'action_horizon_sec'
JPEG_QUALITY = 'jpeg_quality'
SERVER = 'server'

POLICY_META = 'policy'
SERVER_META = f'{POLICY_META}.{SERVER}'
# Recordings on disk carry the policy metadata under either prefix, so a reader accepts both.
POLICY_META_PREFIXES = (POLICY_META, 'inference.policy')

OBS_TIME_NS = 'obs_time_ns'
