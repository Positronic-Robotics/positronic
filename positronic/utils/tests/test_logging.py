import logging
import os

import pytest

from pimm import Sleep, World
from pimm.logging import LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV
from positronic.utils.logging import init_logging

CHILD_LINE = 'init-logging-child-line'
# Between DEBUG and INFO, so a child at this threshold emits `custom_level_loop`'s line and a child
# at INFO does not. The name is registered by the test that wants it, never at import.
CUSTOM_LEVEL, CUSTOM_LEVEL_NAME = 15, 'PIMMTRACE'
# One of the loggers `pimm.logging` pins, named here so the test does not reach for a private tuple.
A_NOISY_LIBRARY = 'websockets'


# Module scope because `start_in_subprocess` pickles the loop.
def logging_loop(stop_reader, clock):
    """A control loop logging one line at INFO, then ending."""
    logging.info(CHILD_LINE)
    yield Sleep(0.001)


def custom_level_loop(stop_reader, clock):
    """A control loop logging one line at a level whose name only the parent registered."""
    logging.log(CUSTOM_LEVEL, CHILD_LINE)
    yield Sleep(0.001)


# What `pimm.world` logs for this loop when it ends the World.
STOP_LINE = f'Stopping background process by {logging_loop.__name__}'


@pytest.fixture(autouse=True)
def _clean_environment_and_root(monkeypatch):
    """`init_logging` writes the environment and reconfigures root; both outlive the test otherwise."""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(RESOLVED_LOG_LEVEL_ENV, raising=False)
    level, handlers = logging.root.level, logging.root.handlers[:]
    noisy = logging.getLogger(A_NOISY_LIBRARY)
    noisy_level = noisy.level
    names, numbers = logging.getLevelNamesMapping(), {**logging._levelToName}  # noqa: SLF001
    yield
    logging.root.handlers[:] = handlers
    logging.root.setLevel(level)
    noisy.setLevel(noisy_level)
    # `monkeypatch.delenv` on a variable that was ABSENT records nothing, so one this test creates
    # outlives it — as the child's threshold, in every test that spawns one after it.
    for variable in (LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV):
        os.environ.pop(variable, None)
    # `addLevelName` writes a process-global registry the stdlib offers no way to unwrite.
    logging._nameToLevel.clear()  # noqa: SLF001
    logging._nameToLevel.update(names)  # noqa: SLF001
    logging._levelToName.clear()  # noqa: SLF001
    logging._levelToName.update(numbers)  # noqa: SLF001


def _stderr_of_a_logging_child(capfd, loop=logging_loop) -> str:
    with World() as world:
        world.start_in_subprocess(loop)
        process = world.background_processes[0]
        process.join(timeout=30)
        assert not process.is_alive(), 'the logging child never exited'
    return capfd.readouterr().err


class TestASecondCallCanStillChangeTheThreshold:
    """A resolved level is `init_logging`'s output, so it must not become its own input next call."""

    def test_the_second_call_wins(self):
        init_logging()
        init_logging('ERROR')

        assert logging.root.level == logging.ERROR
        assert LOG_LEVEL_ENV not in os.environ, "the operator's own variable was written"

    def test_a_child_spawned_after_it_is_suppressed(self, capfd):
        init_logging()
        init_logging('ERROR')

        err = _stderr_of_a_logging_child(capfd)

        assert CHILD_LINE not in err, err
        assert STOP_LINE not in err, err

    def test_the_operators_own_level_outranks_the_argument(self, monkeypatch):
        """`LOG_LEVEL` set by hand beats what a program asks for."""
        monkeypatch.setenv(LOG_LEVEL_ENV, 'DEBUG')

        init_logging('ERROR')

        assert logging.root.level == logging.DEBUG


class TestTheParentConfiguresItselfLikeAChild:
    def test_a_noisy_library_is_pinned_in_the_parent_too(self):
        """The entry point runs the drivers a child runs, so the same libraries are as loud in it."""
        init_logging('DEBUG')

        assert logging.getLogger(A_NOISY_LIBRARY).level == logging.WARNING


class TestALevelTheChildCannotName:
    """The threshold reaches a child as a number, so it does not depend on the child's own registry."""

    def test_a_level_registered_only_in_the_parent_still_starts_the_child(self, capfd):
        # `addLevelName` is process-local and a spawn starts an empty registry: carried as a name,
        # this level would raise in the child and end the World before the loop ran.
        logging.addLevelName(CUSTOM_LEVEL, CUSTOM_LEVEL_NAME)

        init_logging(CUSTOM_LEVEL_NAME)
        err = _stderr_of_a_logging_child(capfd, custom_level_loop)

        assert CHILD_LINE in err, err
        assert STOP_LINE.replace(logging_loop.__name__, custom_level_loop.__name__) in err, err

    def test_a_level_with_no_name_at_all_still_starts_the_child(self, capfd):
        """The same path without `addLevelName`: a number neither process can name is still a level."""
        init_logging(CUSTOM_LEVEL)

        err = _stderr_of_a_logging_child(capfd, custom_level_loop)

        assert CHILD_LINE in err, err

    def test_a_name_that_is_not_a_level_is_still_refused(self):
        """The boundary: carrying numbers must not stop the operator's own typo being rejected."""
        with pytest.raises(ValueError, match='EROR'):
            init_logging('EROR')
