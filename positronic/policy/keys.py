"""Shared policy metadata and observation field names."""

# What a policy reports about itself through its ``meta``; a remote policy nests the server's meta under
# ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy at the top
# level and the vendor under ``SERVER``, so a reader composes a prefix with a field: f'{SERVER_META}.{TYPE}'.
# The block under ``SERVER`` is the server's handshake metadata as sent. A ``prompt`` in it is not the task.
TYPE = 'type'
CHECKPOINT_ID = 'checkpoint_id'
CHECKPOINT_PATH = 'checkpoint_path'
# The license of the weights or the code that a source runs. A remote source states it in its server metadata.
LICENSE = 'license'
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
# The server-issued ID of the session that served the episode.
SESSION_ID = 'session_id'
TRANSCRIPT = 'transcript'
STOP_REASON = 'stop_reason'
HINDSIGHT = 'hindsight'
# The SHA-256 prefix of the API key that an episode drew from a key pool.
API_KEY_FINGERPRINT = 'api_key_fingerprint'

# The reserved field of a result element that holds the ``PolicyEnd`` that ends the run.
END = 'end'

# The record keeps the metadata of each top-level source under f'{SOURCES}.{name}.'.
SOURCES = 'inference'
# The name of the top-level source of the policy of a rollout.
POLICY_SOURCE = 'policy'
POLICY_META = f'{SOURCES}.{POLICY_SOURCE}'
SERVER_META = f'{POLICY_META}.{SERVER}'
# The name of the helper of a run: the top-level source that can take over from the policy.
TELEOPERATOR = 'teleoperator'

# The signals of the control record of a run with a helper.
CONTROL_SOURCE = 'control.source'
CONTROL_USES = 'control.uses'
CONTROL_EVENT = 'control.event'
# The record keeps each shadow action of a top-level source as f'{SHADOW}.{name}.{channel}'.
SHADOW = 'shadow'
# The observation key that the harness adds to the observation of the helper: a ``takeover.ControlObservation`` as
# plain data.
CONTROL = 'control'
# The observation key that holds the ground truth of a simulator: the last values of each privileged signal of the
# harness, by their names in the record. The harness gives it to the helper and to no other source.
GROUND_TRUTH = 'ground_truth'
# The press fields of an element of the helper, beside its commands. A press field holds the time of the last press,
# or ``None`` before the first press, so a new value is a new press.
HELPER_TAKE_OVER = 'take_over'
HELPER_HAND_BACK = 'hand_back'
HELPER_FIELDS = frozenset({HELPER_TAKE_OVER, HELPER_HAND_BACK})

OBS_TIME_NS = 'obs_time_ns'
SAMPLE_INDEX = 'sample_index'
SKIPPED_SAMPLES = 'skipped_samples'
CALL_ID = 'call_id'
CALL_TIMINGS = 'call_timings'
ELAPSED_SEC = 'elapsed_sec'

PLAN = 'plan'
REPLAN_NOW = 'replan_now'


class RTC:
    """Observation fields describing the scheduler's remaining plan and timing."""

    ACTION_PREFIX = 'action_prefix'
    PREFIX_ACTIONS = 'actions'
    CHUNK_OFFSET_SEC = 'chunk_offset_sec'
    OBSERVATION_OFFSET_SEC = 'observation_offset_sec'
