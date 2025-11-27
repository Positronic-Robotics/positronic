import logging
import os

import coloredlogs


def init_logging():
    coloredlogs.install(level='DEBUG')
    log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)-80s',
        datefmt='%H:%M:%S',
        force=True,  # Override any existing configuration
    )
