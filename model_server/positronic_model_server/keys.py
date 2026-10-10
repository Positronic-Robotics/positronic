"""Shared metadata fields reported by model servers."""

from pathlib import Path

MODEL_SETTINGS = 'model_settings'
MODEL_SETTINGS_PATH = Path('meta/positronic_model_settings.json')
ACTION_FPS = 'action_fps'

HOST = 'host'
PORT = 'port'
# The socket path a server bound instead of a host and a port. One of the two pairs is present, never both.
UDS = 'uds'
CHECKPOINT_ID = 'checkpoint_id'
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'
MODEL_SERVER_VERSION = 'model_server_version'
SESSION_PARAMS = 'session_params'
EFFECTIVE_PARAMS = 'effective_params'
