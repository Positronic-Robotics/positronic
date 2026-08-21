"""How a pimm process configures logging, at an entry point and in a spawned control system.

`init_logging` resolves a threshold, publishes it, and configures its own process through
`configure_process_logging`, which is all a spawned child runs. Both therefore take the same
format, threshold and library pins.
"""

import logging
import os
from collections.abc import Mapping
from typing import NamedTuple

import coloredlogs

# A matched pair: `LOG_DATEFMT` renders the `asctime` that `LOG_FORMAT` places, so the two change
# together.
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

    The error names `source`, the variable that carried the name: a threshold that quietly became
    something else reads as working configuration.
    """
    levels = logging.getLevelNamesMapping()
    if name.upper() not in levels:
        raise ValueError(f'{source}={name!r} is not a logging level (one of {", ".join(sorted(levels))})')
    return levels[name.upper()]


def _requested_level() -> int:
    """The threshold: the level a parent resolved, else the operator's own, else INFO."""
    # An empty variable is a value the operator set, and `init_logging` raises on it: read as INFO
    # here, one operator input would mean two things across a spawn.
    resolved = os.getenv(RESOLVED_LOG_LEVEL_ENV)
    if resolved is not None:
        # Ours to write and ours to read, so a value that is not a number is a broken handoff rather
        # than an operator's typo. It raises either way; the message says which.
        if not resolved.lstrip('-').isdigit():
            raise ValueError(f'{RESOLVED_LOG_LEVEL_ENV}={resolved!r} is not a numeric logging level')
        return int(resolved)
    name = os.getenv(LOG_LEVEL_ENV)
    return level_number(name, LOG_LEVEL_ENV) if name is not None else logging.INFO


def component_log_levels() -> dict[str, int]:
    """Every logger this process has a level of its own, for a spawn to carry.

    The root logger is absent: its level is the threshold, which crosses as `RESOLVED_LOG_LEVEL_ENV`.
    """
    return {
        name: logger.level
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET
    }


class _Pin(NamedTuple):
    """A noisy library's two levels: the application's own, and what this module installed over it."""

    theirs: int
    ours: int


# Both levels, because a pin often equals the application's own: told apart by value alone, a pin
# would survive as a setting and a second `init_logging` could not lower it.
_pins: dict[str, _Pin] = {}


def configure_process_logging(component_levels: Mapping[str, int] | None = None) -> None:
    """Configure this process's root logger, per-component levels and library pins.

    A process nothing else configures — a spawned control system — would otherwise sit at the stdlib
    default and drop every line it emits. `component_levels` is what the spawning process had set
    per logger, which a fresh interpreter starts with none of.

    Raises `ValueError` on a value that names no level.
    """
    level = _requested_level()
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    # An ancestor's level does not filter a record its own logger admitted — only a handler's does,
    # and `basicConfig` leaves that at NOTSET.
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
    # Before the pins, so a noisy library the spawning application had raised is read below as that
    # application's setting rather than as one of ours.
    for name, component_level in (component_levels or {}).items():
        logging.getLogger(name).setLevel(component_level)
    for logger_name in _NOISY_LIBRARY_LOGGERS:
        logger = logging.getLogger(logger_name)
        pinned = _pins.get(logger_name)
        # A level this module did not install is the application's own; under one it did, what it
        # recorded is. NOTSET is 0, so a library nobody set takes the floor.
        theirs = pinned.theirs if pinned is not None and logger.level == pinned.ours else logger.level
        pin = max(level, logging.WARNING, theirs)
        logger.setLevel(pin)
        _pins[logger_name] = _Pin(theirs, pin)


def init_logging(level: str | int = 'INFO') -> None:
    """Configure the entry point's own process, and publish the threshold its children read.

    `level` is what the program asks for; the operator's own `LOG_LEVEL` outranks it. Colour is the
    one thing an entry point does that a child does not: a child's output is a pipe.
    """
    requested = os.getenv(LOG_LEVEL_ENV)
    if requested is not None:
        log_level = level_number(requested, LOG_LEVEL_ENV)
    else:
        log_level = level_number(level, 'level') if isinstance(level, str) else level

    # A number, and in its own variable: a name resolves against the reading process's registry, and
    # `LOG_LEVEL` is the operator's, so writing there makes this call's output its own next input.
    os.environ[RESOLVED_LOG_LEVEL_ENV] = str(log_level)
    configure_process_logging()  # reads the number just written, so a parent configures as its children do
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
