import logging
import os

import coloredlogs

from pimm.logging import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV, level_number


def init_logging(level: str | int = 'INFO'):
    requested = os.getenv(LOG_LEVEL_ENV)  # the operator's own, and it outranks what a caller asked for
    if requested is not None:
        log_level = level_number(requested, LOG_LEVEL_ENV)
    else:
        log_level = level_number(level, 'level') if isinstance(level, str) else level

    # Spawned children carry no logging config, so they read the resolved level back
    # (`pimm.logging.configure_process_logging`). It travels as a number: a name is resolved against
    # the reading process's registry, and one registered here by `addLevelName` names nothing in a
    # freshly spawned interpreter. It goes in its own variable rather than in `LOG_LEVEL`, which is
    # the operator's: written there it would be this function's own input on the next call, and a
    # later `init_logging('ERROR')` would read back the level the first call resolved.
    os.environ[RESOLVED_LOG_LEVEL_ENV] = str(log_level)
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,  # Override any existing configuration
    )
    coloredlogs.install(level=log_level, fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
