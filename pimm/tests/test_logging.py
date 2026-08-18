import contextlib
import logging

import pimm
from pimm.utils import RateCounter

# What the two components below emit at INFO, and what the assertions look for.
COUNTER_PREFIX = 'level-probe'
WORLD_LINE = 'Stopping background processes...'


@contextlib.contextmanager
def _level(name: str, level: int):
    """One logger's level for the block, put back after. `caplog.set_level` also moves its own
    handler, which would drop the records the assertions read."""
    logger = logging.getLogger(name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _emit_one_info_line_from_each_component() -> None:
    """One INFO record from `pimm.utils`, one from `pimm.world`."""
    counter = RateCounter(COUNTER_PREFIX, level=logging.INFO)
    counter.tick()
    counter.report()
    with pimm.World():
        pass


def _counter_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(COUNTER_PREFIX)]


def _world_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage() == WORLD_LINE]


class TestPerComponentLevels:
    """Each module logs under its own name, so an operator can move one component's threshold alone.

    Under the root logger there was one threshold for everything: the first two assertions below could
    not both hold, whichever way it was set.
    """

    def test_raising_one_components_level_silences_that_component(self, caplog):
        caplog.set_level(logging.INFO)

        with _level('pimm.utils', logging.WARNING):
            _emit_one_info_line_from_each_component()

        assert _counter_lines(caplog) == [], caplog.text

    def test_raising_one_components_level_leaves_the_others_alone(self, caplog):
        caplog.set_level(logging.INFO)

        with _level('pimm.utils', logging.WARNING):
            _emit_one_info_line_from_each_component()

        # The boundary: a threshold that reached past `pimm.utils` would take this line with it.
        assert _world_lines(caplog) == [WORLD_LINE], caplog.text

    def test_both_components_log_at_a_shared_threshold(self, caplog):
        """The control: with nothing pinned both lines land, so the silence above is the pin rather
        than a component that never logged."""
        caplog.set_level(logging.INFO)

        _emit_one_info_line_from_each_component()

        assert len(_counter_lines(caplog)) == 1, caplog.text
        assert _world_lines(caplog) == [WORLD_LINE], caplog.text
