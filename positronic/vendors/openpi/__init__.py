"""What positronic states about OpenPI: the observation its subprocess takes, and the assets it needs on hand."""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# One name per field the input transforms pull out. Which subset a checkpoint reads follows the config it is
# served under: the ``pi05_positronic`` and ``pi05_libero`` transforms take the concatenated ``STATE`` and two
# cameras, the ``pi05_droid`` ones take joints and gripper apart with their own camera names (openpi
# ``positronic_policy``, ``libero_policy`` and ``droid_policy``).
STATE = 'observation/state'
IMAGE = 'observation/image'
WRIST_IMAGE = 'observation/wrist_image'

JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
EXTERIOR_IMAGE_LEFT = 'observation/exterior_image_1_left'
WRIST_IMAGE_LEFT = 'observation/wrist_image_left'

PROMPT = 'prompt'

# The one ``STATE`` width every config accepts: the transform that does not pad takes exactly this, and the
# ones that do pad it up to the action dimension. What a codec encodes is wider or narrower, but never
# reaches the model as either — padding runs first, so the width a warmup sends is not observable to it.
_STATE_DIM = 8


def warm_observation() -> dict[str, Any]:
    """Zero-filled inputs carrying every field above, so one observation warms a checkpoint under any config.

    A config's input transform pulls out the fields it names and ignores the rest, which is what lets one
    observation stand in for the codec that normally feeds the subprocess. Images are resized by openpi itself.
    """
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        STATE: np.zeros(_STATE_DIM, dtype=np.float32),
        IMAGE: frame,
        WRIST_IMAGE: frame,
        JOINT_POSITION: np.zeros(7, dtype=np.float32),
        GRIPPER_POSITION: np.zeros(1, dtype=np.float32),
        EXTERIOR_IMAGE_LEFT: frame,
        WRIST_IMAGE_LEFT: frame,
        PROMPT: '',
    }


# Google revoked anonymous access to gs://big_vision, breaking OpenPI's tokenizer download.
# Track: https://github.com/Physical-Intelligence/openpi/issues/881
_TOKENIZER_URL = 'https://storage.eu-north1.nebius.cloud/positronic-public/assets/paligemma_tokenizer.model'
# Mirror upstream openpi's OPENPI_DATA_HOME convention so the tokenizer co-locates
# with the gs://openpi-assets cache (default ~/.cache/openpi when unset).
_OPENPI_DATA_HOME = Path(os.getenv('OPENPI_DATA_HOME', Path.home() / '.cache' / 'openpi')).expanduser()
_TOKENIZER_CACHE = _OPENPI_DATA_HOME / 'big_vision' / 'paligemma_tokenizer.model'


def ensure_paligemma_tokenizer():
    """Download PaliGemma tokenizer if not already cached."""
    if _TOKENIZER_CACHE.exists():
        return
    logger.info(f'Downloading PaliGemma tokenizer to {_TOKENIZER_CACHE}')
    _TOKENIZER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_TOKENIZER_URL, _TOKENIZER_CACHE)
