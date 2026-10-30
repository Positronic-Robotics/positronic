"""Shared policy metadata and observation field names."""

# What a policy reports about itself through its ``meta``; a remote policy nests the server's meta under
# ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy at the top
# level and the vendor under ``SERVER``, so a reader composes a prefix with a field: f'{SERVER_META}.{TYPE}'.
# The block under ``SERVER`` is the server's handshake metadata as sent. A ``prompt`` in it is not the task.
TYPE = 'type'
CHECKPOINT_ID = 'checkpoint_id'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
# The `provider:model` id that an API policy calls.
MODEL = 'model'
# The `TYPE` of an LLM policy, which the results tables name by its `MODEL`.
LLM_TYPE = 'llm'
ACTION_HORIZON_SEC = 'action_horizon_sec'
ACTION_FPS = 'action_fps'
JPEG_QUALITY = 'jpeg_quality'
# Metadata understood by the legacy offboard client and server.
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'
# The name of the endpoint that served the episode.
LABEL = 'label'
SERVER = 'server'
TRANSCRIPT = 'transcript'
STOP_REASON = 'stop_reason'
HINDSIGHT = 'hindsight'
# The SHA-256 prefix of the API key that an episode drew from a key pool.
API_KEY_FINGERPRINT = 'api_key_fingerprint'

# The reserved field of a result element that holds the ``PolicyEnd`` that ends the run.
END = 'end'

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'

OBS_TIME_NS = 'obs_time_ns'


class RTC:
    """Observation fields describing the scheduler's remaining plan and timing."""

    ACTION_PREFIX = 'action_prefix'
    PREFIX_ACTIONS = 'actions'
    CHUNK_OFFSET_SEC = 'chunk_offset_sec'
    OBSERVATION_OFFSET_SEC = 'observation_offset_sec'
    ESTIMATED_LATENCY_SEC = 'estimated_latency_sec'
    MEASURED_LATENCY_SEC = 'measured_latency_sec'
