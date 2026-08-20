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
# The level a parent resolved, for spawned children — a NUMBER, because a name resolves against the
# reading process's registry and a spawn starts an empty one. Unset, a child logs at INFO.
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


def level_number(name: str, source: str) -> int:
    """The number `name` stands for in this process, raising when it names no level.

    `source` is what carried the name, and naming it is most of the error's value: a threshold that
    quietly became something else reads as working configuration.
    """
    levels = logging.getLevelNamesMapping()
    if name.upper() not in levels:
        raise ValueError(f'{source}={name!r} is not a logging level (one of {", ".join(sorted(levels))})')
    return levels[name.upper()]


def _requested_level() -> int:
    """The threshold: the level a parent resolved, else the operator's own, else INFO."""
    resolved = os.getenv(RESOLVED_LOG_LEVEL_ENV)
    if resolved:
        # Ours to write and ours to read, so a value that is not a number is a broken handoff rather
        # than an operator's typo. It raises either way; the message says which.
        if not resolved.lstrip('-').isdigit():
            raise ValueError(f'{RESOLVED_LOG_LEVEL_ENV}={resolved!r} is not a numeric logging level')
        return int(resolved)
    name = os.getenv(LOG_LEVEL_ENV)
    return level_number(name, LOG_LEVEL_ENV) if name else logging.INFO


def configure_process_logging() -> None:
    """Configure this process's root logger and its library pins from the environment.

    A process nothing else configures — a spawned control system — would otherwise sit at the stdlib
    default and drop every line it emits. The threshold is the level a parent resolved, else the
    operator's own, so a requested suppression reaches a control system rather than stopping at the
    parent; the noisy libraries are pinned no lower than WARNING.

    Raises `ValueError` on a value that names no level, which would otherwise configure some other
    threshold and read as working configuration.
    """
    level = _requested_level()
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    for logger_name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))
