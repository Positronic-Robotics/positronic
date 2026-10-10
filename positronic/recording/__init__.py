"""Timeline conventions for robot recordings and their consumers."""

from collections.abc import Iterable

from pimm.time import RECEIVED_WORLD, validate_timeline
from positronic.dataset.vector import RECORDED_TIME

HARNESS_WORLD = 'harness.world'
HARNESS_WALL = 'harness.wall'


def select_timeline(timelines: Iterable[str], *, timeline: str | None = None) -> str:
    """Select an explicit timeline or prefer Harness, receipt, then legacy recorded world time."""
    available = set(timelines)
    if timeline is not None:
        validate_timeline(timeline)
        if timeline not in available:
            raise KeyError(timeline)
        return timeline
    for name in (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME):
        if name in available:
            return name
    raise ValueError(f'Select an explicit timeline from {sorted(available)}')
