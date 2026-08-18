import logging
import os

import coloredlogs

from pimm.logging import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV


def init_logging(level: str | int = 'INFO'):
    if isinstance(level, int):
        level = logging.getLevelName(level)

    # Default to the passed level, but allow env var override
    log_level = os.getenv(LOG_LEVEL_ENV, level).upper()
    # Spawned children carry no logging config, so they read the resolved level back
    # (`pimm.logging.configure_process_logging`). It goes in its own variable rather than in
    # `LOG_LEVEL`, which is the operator's: written there it would be this function's own input on the
    # next call, and a later `init_logging('ERROR')` would read back the level the first call resolved.
    os.environ[RESOLVED_LOG_LEVEL_ENV] = log_level
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,  # Override any existing configuration
    )
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
