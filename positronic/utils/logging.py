import logging
import os

import coloredlogs

from pimm.logging import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL_ENV


def init_logging(level: str | int = 'INFO'):
    if isinstance(level, int):
        level = logging.getLevelName(level)

    # Default to the passed level, but allow env var override
    log_level = os.getenv(LOG_LEVEL_ENV, level).upper()
    # Spawned children carry no logging config, so they read this back (`pimm.world._init_child_logging`).
    os.environ[LOG_LEVEL_ENV] = log_level
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,  # Override any existing configuration
    )
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
