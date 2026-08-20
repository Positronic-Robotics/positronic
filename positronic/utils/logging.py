import os

import coloredlogs

from pimm.logging import (
    LOG_DATEFMT,
    LOG_FORMAT,
    LOG_LEVEL_ENV,
    RESOLVED_LOG_LEVEL_ENV,
    configure_process_logging,
    level_number,
)


def init_logging(level: str | int = 'INFO'):
    """Configure the entry point's own process, and publish the level its children read."""
    requested = os.getenv(LOG_LEVEL_ENV)  # the operator's own, and it outranks what a caller asked for
    if requested is not None:
        log_level = level_number(requested, LOG_LEVEL_ENV)
    else:
        log_level = level_number(level, 'level') if isinstance(level, str) else level

    # A number, and in its own variable: a name resolves against the reading process's registry, and
    # `LOG_LEVEL` is the operator's, so writing there makes this call's output its own next input.
    os.environ[RESOLVED_LOG_LEVEL_ENV] = str(log_level)
    configure_process_logging()  # reads the number just written, so a parent configures as its children do
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
