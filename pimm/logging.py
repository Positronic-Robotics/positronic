"""How a pimm process configures logging — the parent's format, and a spawned child's own setup.

A spawned control system runs no entry point, so nothing else configures it; `init_child_logging`
is what gives it a threshold and the shared format. The parent's own configuration is
`positronic.utils.logging`, which reads its format from here.
"""

import logging
import os

# A matched pair: `LOG_DATEFMT` renders the `asctime` that `LOG_FORMAT` places, so the two change
# together. Public — `positronic.utils.logging` configures the parent from them.
LOG_FORMAT = '%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)-80s'
LOG_DATEFMT = '%H:%M:%S'

# The threshold's one carrier between a parent and its spawned children: spawn passes no logging
# configuration, so the parent writes the level it resolved here and a child reads it back.
LOG_LEVEL_ENV = 'LOG_LEVEL'

# Third-party loggers the child's root-level INFO would otherwise reach too. Each logs per connection,
# request or retry rather than per event, so pinning them keeps the level change ours.
_NOISY_LIBRARY_LOGGERS = (
    'websockets',  # per connection at INFO, per frame at DEBUG
    'httpx',  # per request, at INFO
    'httpcore',  # per connection-pool operation
    'urllib3',  # per connection
    'botocore',  # per API call, and again per retry
    'boto3',
    's3transfer',  # per part of a multipart upload
    'asyncio',  # per selector event, under its debug mode
)


def init_child_logging() -> None:
    """Configure a spawned child's own logging from `LOG_LEVEL`, defaulting to INFO.

    A spawned child runs no entry point, so nothing else configures it and its root logger would
    otherwise sit at the stdlib default, dropping every line a control system emits. `LOG_LEVEL` is
    what the parent resolves its own threshold from (`positronic.utils.logging.init_logging` writes
    the level it resolved back into the environment), so a suppression reaches a control system
    rather than stopping at the parent. The noisy libraries are pinned no lower than WARNING.
    """
    level = logging.getLevelNamesMapping().get(os.getenv(LOG_LEVEL_ENV, 'INFO').upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    for name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
