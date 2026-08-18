import logging
import os

import coloredlogs

from pimm.world import LOG_DATEFMT, LOG_FORMAT


def init_logging(level: str | int = 'INFO'):
    if isinstance(level, int):
        level = logging.getLevelName(level)

    # Default to the passed level, but allow env var override
    log_level = os.getenv('LOG_LEVEL', level).upper()
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,  # Override any existing configuration
    )
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
