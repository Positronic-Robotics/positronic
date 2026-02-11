import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Google revoked anonymous access to gs://big_vision, breaking OpenPI's tokenizer download.
# Track: https://github.com/Physical-Intelligence/openpi/issues/881
_TOKENIZER_URL = 'https://storage.eu-north1.nebius.cloud/positronic-public/assets/paligemma_tokenizer.model'
_TOKENIZER_CACHE = Path.home() / '.cache' / 'openpi' / 'big_vision' / 'paligemma_tokenizer.model'


def ensure_paligemma_tokenizer():
    """Download PaliGemma tokenizer if not already cached."""
    if _TOKENIZER_CACHE.exists():
        return
    logger.info(f'Downloading PaliGemma tokenizer to {_TOKENIZER_CACHE}')
    _TOKENIZER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_TOKENIZER_URL, _TOKENIZER_CACHE)
