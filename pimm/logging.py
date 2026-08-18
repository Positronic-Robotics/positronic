"""How a pimm process configures logging — the parent's format, and a spawned child's own setup.

A spawned control system runs no entry point, so nothing else configures it;
`configure_process_logging` is what gives it a threshold and the shared format. The parent's own
configuration is `positronic.utils.logging`, which reads its format from here.
"""

import logging
import os

# A matched pair: `LOG_DATEFMT` renders the `asctime` that `LOG_FORMAT` places, so the two change
# together. Public — `positronic.utils.logging` configures the parent from them.
LOG_FORMAT = '%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)-80s'
LOG_DATEFMT = '%H:%M:%S'

# The operator's own input, read and never written: a resolved level stored here would be the next
# `init_logging` call's own input, so a later `init_logging('ERROR')` could not change the threshold.
LOG_LEVEL_ENV = 'LOG_LEVEL'
# The level a parent resolved, for spawned children, which carry no logging configuration of their
# own. Unset means nothing configured a threshold, and a child then logs at INFO rather than falling
# silent — an entry point is what asks for a threshold and a spawned control system has none.
RESOLVED_LOG_LEVEL_ENV = 'PIMM_RESOLVED_LOG_LEVEL'

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


def _requested_level() -> tuple[str, str]:
    """The level name to configure with, and the variable that named it."""
    for variable in (RESOLVED_LOG_LEVEL_ENV, LOG_LEVEL_ENV):
        value = os.getenv(variable)
        if value:
            return variable, value.upper()
    return RESOLVED_LOG_LEVEL_ENV, 'INFO'


def configure_process_logging() -> None:
    """Configure this process's root logger and its library pins from the environment.

    A process nothing else configures — a spawned control system — would otherwise sit at the stdlib
    default and drop every line it emits. The threshold is the level a parent resolved, else the
    operator's own, so a requested suppression reaches a control system rather than stopping at the
    parent; the noisy libraries are pinned no lower than WARNING.

    Raises `ValueError` on a value that is not a level, which would otherwise configure some other
    threshold and read as working configuration.
    """
    variable, name = _requested_level()
    levels = logging.getLevelNamesMapping()
    if name not in levels:
        raise ValueError(f'{variable}={name!r} is not a logging level (one of {", ".join(sorted(levels))})')
    level = levels[name]
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    for logger_name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))
