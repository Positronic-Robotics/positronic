"""The timelines that a recording keeps as time columns, and the order in which a reader picks one.

pimm and the harness write these names, and a labeler and a dataset reader read them, so all import them from here.
"""

from collections.abc import Container

EMITTED_PREFIX = 'emitted.'
RECEIVED_PREFIX = 'received.'
EMITTED_WALL = f'{EMITTED_PREFIX}wall'
# The timeline of when each sample was made, on the world clock that a trial runs on.
EMITTED_WORLD = f'{EMITTED_PREFIX}world'
RECEIVED_WALL = f'{RECEIVED_PREFIX}wall'
RECEIVED_WORLD = f'{RECEIVED_PREFIX}world'
# The harness's time of each message it records: the first receipt of an input, the emission of a command.
HARNESS_WALL = 'harness.wall'
HARNESS_WORLD = 'harness.world'
# The timeline of a signal file from before named timelines, where the file names no timeline of its own.
RECORDED_TIME = 'recorded'

# The timelines that a reader reads a recording's sample times from, in the order that it tries them.
# TODO: Remove the received.world fallback when teleoperation becomes a Harness policy and `data_collection.py`
# is removed.
RECORDING_TIMELINE_ORDER = (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME)


def recording_timeline(timelines: Container[str]) -> str:
    """The first of `RECORDING_TIMELINE_ORDER` in `timelines`, else `RECORDED_TIME`."""
    return next((name for name in RECORDING_TIMELINE_ORDER if name in timelines), RECORDED_TIME)
