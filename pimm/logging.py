"""Logging for a pimm program, whose control systems may run in a subprocess of their own.

Control systems are spawned rather than forked, so a subprocess starts a fresh interpreter holding
none of the main process's logging configuration, and the stdlib default drops most of what a
control system emits. This module carries the configuration across that boundary, so one
`LOG_LEVEL` in the environment covers the main process and every subprocess it spawns.

`init_logging` is the one call an application makes: it resolves the threshold and leaves it in the
environment subprocesses inherit. `configure_process_logging` then configures every process —
`init_logging` runs it on the main process, and a spawned subprocess runs it before its control
system starts, with the per-logger levels `component_log_levels` collected before the spawn.
"""

import logging
import os
from collections.abc import Mapping
from typing import NamedTuple

import coloredlogs

LOG_FORMAT = '%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)-80s'
LOG_DATEFMT = '%H:%M:%S'  # renders the `asctime` in `LOG_FORMAT`, so changing either alone breaks the timestamp

# The environment variable that sets the threshold for the whole program. It is read-only: it takes
# precedence over `init_logging`'s `level`, so a value written back would beat the argument of every
# later call.
LOG_LEVEL_ENV = 'LOG_LEVEL'
# How the main process hands the resolved threshold to a subprocess. It carries a number, not a
# level name: names resolve through `logging.getLevelNamesMapping()`, which is per-process, and a
# fresh interpreter holds only the standard ones — anything `addLevelName` added is gone.
RESOLVED_LOG_LEVEL_ENV = 'PIMM_RESOLVED_LOG_LEVEL'
# These libraries log per connection, request or retry, so a lowered threshold would flood the output.
_NOISY_LIBRARY_LOGGERS = (
    'websockets',  # per connection at INFO, per frame at DEBUG
    'httpx',  # per request, at INFO
    'httpcore',  # per connection-pool operation
    'urllib3',  # per connection
    'botocore',  # per API call, and again per retry
    'boto3',
    's3transfer',  # per part of a multipart upload
    'asyncio',  # per selector event, under its debug mode
    'linuxpy',  # per ioctl, which is several times a frame for every camera
)


def level_number(level_name: str, source: str) -> int:
    """The number `level_name` stands for in this process, raising when it names no level."""
    levels = logging.getLevelNamesMapping()
    if level_name.upper() not in levels:
        raise ValueError(f'{source}={level_name!r} is not a logging level (one of {", ".join(sorted(levels))})')
    return levels[level_name.upper()]


def _requested_level() -> int:
    """The level this process logs at."""
    # An empty `LOG_LEVEL` or `PIMM_RESOLVED_LOG_LEVEL` raises rather than falling back to INFO.
    resolved = os.getenv(RESOLVED_LOG_LEVEL_ENV)
    if resolved is not None:
        if not resolved.lstrip('-').isdigit():
            raise ValueError(f'{RESOLVED_LOG_LEVEL_ENV}={resolved!r} is not a numeric logging level')
        return int(resolved)
    level_name = os.getenv(LOG_LEVEL_ENV)
    return level_number(level_name, LOG_LEVEL_ENV) if level_name is not None else logging.INFO


def component_log_levels() -> dict[str, int]:
    """The level of every logger this process set one on, for a spawn to carry.

    The root logger is absent: its level is the threshold, which crosses as `RESOLVED_LOG_LEVEL_ENV`.
    """
    return {
        name: logger.level
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET
    }


class _Pin(NamedTuple):
    """A noisy library's two levels: the application's own, and what this module installed over it.

    A pin often equals the level the application itself set. Recording only the pin would read that
    setting as this module's own, and a second `init_logging` could not lower it.
    """

    theirs: int
    ours: int


_pins: dict[str, _Pin] = {}


def configure_process_logging(component_levels: Mapping[str, int] | None = None) -> None:
    """Configure this process's root logger, per-component levels and library pins.

    `component_levels` holds the per-logger levels the main process had set. A fresh interpreter has
    none of them, so a subprocess is given them here.

    Raises `ValueError` on a value that names no level.
    """
    level = _requested_level()
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    # A component logger admits a record on its own level. The root logger's level is not consulted
    # after that, so only its handler's level can still filter the record, and `basicConfig` leaves
    # that at NOTSET.
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
    # The application's own levels go on first. The pin loop below reads each library logger's
    # current level to work out what it asked for.
    for name, component_level in (component_levels or {}).items():
        logging.getLogger(name).setLevel(component_level)
    for logger_name in _NOISY_LIBRARY_LOGGERS:
        logger = logging.getLogger(logger_name)
        pinned = _pins.get(logger_name)
        theirs = pinned.theirs if pinned is not None and logger.level == pinned.ours else logger.level
        pin = max(level, logging.WARNING, theirs)  # NOTSET is 0, so a library nobody set takes the floor
        logger.setLevel(pin)
        _pins[logger_name] = _Pin(theirs, pin)


def init_logging(level: str | int = 'INFO') -> None:
    """Configure the main process and publish the threshold its subprocesses read.

    `level` is what the program asks for, and `LOG_LEVEL` takes precedence over it. The main process
    also gets colour, which a subprocess does not: its output is a pipe.
    """
    requested = os.getenv(LOG_LEVEL_ENV)
    if requested is not None:
        log_level = level_number(requested, LOG_LEVEL_ENV)
    else:
        log_level = level_number(level, 'level') if isinstance(level, str) else level

    os.environ[RESOLVED_LOG_LEVEL_ENV] = str(log_level)
    configure_process_logging()  # reads the number just written, so the main process configures as its subprocesses do
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
