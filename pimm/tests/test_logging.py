import contextlib
import logging

import pytest

import pimm
from pimm.logging import LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV, configure_process_logging
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


class TestRequestedLevel:
    """What `configure_process_logging` reads, and what it refuses."""

    @pytest.fixture(autouse=True)
    def _restore_root(self):
        """`configure_process_logging` reconfigures the root logger, which every later test shares."""
        level, handlers = logging.root.level, logging.root.handlers[:]
        yield
        logging.root.handlers[:] = handlers
        logging.root.setLevel(level)

    @pytest.mark.parametrize('variable', [LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV])
    def test_a_value_that_is_not_a_level_is_refused(self, monkeypatch, variable):
        monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
        monkeypatch.delenv(RESOLVED_LOG_LEVEL_ENV, raising=False)
        monkeypatch.setenv(variable, 'EROR')

        # Naming the variable is the point: a typo silently read as INFO looks like working
        # configuration, and the parent would have raised on the same value.
        with pytest.raises(ValueError, match=f'{variable}=.EROR.'):
            configure_process_logging()

    def test_a_number_this_process_cannot_name_is_a_level(self, monkeypatch):
        """The boundary against the refusal above: an unnamed number is what a parent's custom level
        arrives as, so refusing it would end a World over configuration that is entirely valid."""
        monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
        monkeypatch.setenv(RESOLVED_LOG_LEVEL_ENV, '15')
        assert 15 not in logging.getLevelNamesMapping().values(), 'pick a number this process cannot name'

        configure_process_logging()

        assert logging.root.level == 15

    def test_the_resolved_level_outranks_the_operators_own(self, monkeypatch):
        """`init_logging` resolved its level having already read `LOG_LEVEL`, so it is the informed one."""
        monkeypatch.setenv(LOG_LEVEL_ENV, 'DEBUG')
        monkeypatch.setenv(RESOLVED_LOG_LEVEL_ENV, str(logging.ERROR))

        configure_process_logging()

        assert logging.root.level == logging.ERROR

    def test_an_unconfigured_parent_leaves_this_process_at_info(self, monkeypatch):
        """Neither variable set means nothing asked for a threshold, which must not read as silence."""
        monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
        monkeypatch.delenv(RESOLVED_LOG_LEVEL_ENV, raising=False)

        configure_process_logging()

        assert logging.root.level == logging.INFO


class TestPerComponentLevels:
    """Each module logs under its own name, so a threshold set on one component moves only that one."""

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
