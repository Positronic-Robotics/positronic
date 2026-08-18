import logging
import os

import pytest

from pimm import Sleep, World
from pimm.logging import LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV
from positronic.utils.logging import init_logging

CHILD_LINE = 'init-logging-child-line'


# Module scope because `start_in_subprocess` pickles the loop.
def logging_loop(stop_reader, clock):
    """A control loop logging one line at INFO, then ending."""
    logging.info(CHILD_LINE)
    yield Sleep(0.001)


# What `pimm.world` logs for this loop when it ends the World.
STOP_LINE = f'Stopping background process by {logging_loop.__name__}'


@pytest.fixture(autouse=True)
def _clean_environment_and_root(monkeypatch):
    """`init_logging` writes the environment and reconfigures root; both outlive the test otherwise."""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(RESOLVED_LOG_LEVEL_ENV, raising=False)
    level, handlers = logging.root.level, logging.root.handlers[:]
    yield
    logging.root.handlers[:] = handlers
    logging.root.setLevel(level)


def _stderr_of_a_logging_child(capfd) -> str:
    with World() as world:
        world.start_in_subprocess(logging_loop)
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

    def test_the_operators_own_level_still_outranks_the_argument(self, monkeypatch):
        """Unchanged precedence: `LOG_LEVEL` set by hand beats what a program asks for."""
        monkeypatch.setenv(LOG_LEVEL_ENV, 'DEBUG')

        init_logging('ERROR')

        assert logging.root.level == logging.DEBUG
