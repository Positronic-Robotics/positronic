import logging
import os

import coloredlogs


def init_logging(level: str | int = 'INFO'):
    if isinstance(level, int):
        level = logging.getLevelName(level)

    # Default to the passed level, but allow env var override
    log_level = os.getenv('LOG_LEVEL', level).upper()
    fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)-80s'
    datefmt = '%H:%M:%S'
    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        force=True,  # Override any existing configuration
    )
    coloredlogs.install(level=log_level, fmt=fmt, datefmt=datefmt)
