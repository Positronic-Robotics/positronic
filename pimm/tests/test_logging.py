import contextlib
import logging
import os

import pytest

import pimm
import pimm.logging
from pimm.logging import LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV, configure_process_logging, init_logging
from pimm.utils import RateCounter

# What `pimm.utils` and `pimm.world` emit at INFO, and what the assertions look for.
COUNTER_PREFIX = 'level-probe'
WORLD_LINE = 'Stopping background processes...'

CHILD_LINE = 'init-logging-child-line'
# `CUSTOM_LEVEL` sits between DEBUG and INFO, so a child at this threshold emits
# `custom_level_loop`'s line and a child at INFO does not. `CUSTOM_LEVEL_NAME` is registered by the
# test that wants it, never at import.
CUSTOM_LEVEL, CUSTOM_LEVEL_NAME = 15, 'PIMMTRACE'
# One of the loggers `pimm.logging` pins, named here so the test does not reach for a private tuple.
A_NOISY_LIBRARY = 'websockets'
# Stands for a module that sets its own level at import, as `positronic.dataset.ds_writer_agent`
# does: its records clear its own logger, so only a handler's threshold can stop them.
A_SELF_LEVELLED_COMPONENT = 'pimm.tests.a-self-levelled-component'
# Stands for a driver addressed by name, logged from a child under that same name.
A_COMPONENT = 'pimm.tests.a-component'


# Module scope because `start_in_subprocess` pickles the loop.
def logging_loop(stop_reader, clock):
    """A control loop logging one line at INFO, then ending."""
    logging.info(CHILD_LINE)
    yield pimm.Sleep(0.001)


def custom_level_loop(stop_reader, clock):
    """A control loop logging one line at a level whose name only the parent registered."""
    logging.log(CUSTOM_LEVEL, CHILD_LINE)
    yield pimm.Sleep(0.001)


def component_loop(stop_reader, clock):
    """A control loop logging one line at INFO under a component logger of its own."""
    logging.getLogger(A_COMPONENT).info(CHILD_LINE)
    yield pimm.Sleep(0.001)


# What `pimm.world` logs for this loop when it ends the World.
STOP_LINE = f'Stopping background process by {logging_loop.__name__}'


@pytest.fixture(autouse=True)
def _clean_environment_and_root(monkeypatch):
    """Put back what `init_logging` and `configure_process_logging` change: the environment
    variables, the root logger, the library pins and the level-name registry."""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(RESOLVED_LOG_LEVEL_ENV, raising=False)
    level, handlers = logging.root.level, logging.root.handlers[:]
    noisy = logging.getLogger(A_NOISY_LIBRARY)
    noisy_level = noisy.level
    names, numbers = logging.getLevelNamesMapping(), {**logging._levelToName}
    pimm.logging._pins.clear()
    yield
    pimm.logging._pins.clear()
    logging.root.handlers[:] = handlers
    logging.root.setLevel(level)
    noisy.setLevel(noisy_level)
    # `monkeypatch.delenv` on a variable that was ABSENT records nothing, so one this test creates
    # outlives it — as the child's threshold, in every test that spawns one after it.
    for variable in (LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV):
        os.environ.pop(variable, None)
    # `addLevelName` writes a process-global registry the stdlib offers no way to unwrite.
    logging._nameToLevel.clear()
    logging._nameToLevel.update(names)
    logging._levelToName.clear()
    logging._levelToName.update(numbers)


@contextlib.contextmanager
def _level(name: str, level: int):
    """One logger's level for the block, put back after.

    `caplog.set_level` would do the same but also moves its own handler, dropping the records the
    assertions read.
    """
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


def _stderr_of_a_logging_child(capfd, loop=logging_loop) -> str:
    with pimm.World() as world:
        world.start_in_subprocess(loop)
        process = world.background_processes[0]
        process.join(timeout=30)
        assert not process.is_alive(), 'the logging child never exited'
    return capfd.readouterr().err


class TestRequestedLevel:
    """What `configure_process_logging` reads, and what it refuses."""

    @pytest.mark.parametrize('variable', [LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV])
    def test_a_value_that_is_not_a_level_is_refused(self, monkeypatch, variable):
        monkeypatch.setenv(variable, 'EROR')

        # Naming the variable is the point: a typo silently read as INFO looks like working
        # configuration, and the parent would have raised on the same value.
        with pytest.raises(ValueError, match=f'{variable}=.EROR.'):
            configure_process_logging()

    def test_a_number_this_process_cannot_name_is_a_level(self, monkeypatch):
        """A parent's custom level arrives as a number this process cannot name, so refusing it
        would end a World over valid configuration."""
        monkeypatch.setenv(RESOLVED_LOG_LEVEL_ENV, str(CUSTOM_LEVEL))
        assert CUSTOM_LEVEL not in logging.getLevelNamesMapping().values(), 'pick an unnameable number'

        configure_process_logging()

        assert logging.root.level == CUSTOM_LEVEL

    def test_the_resolved_level_takes_precedence_over_log_level(self, monkeypatch):
        """`init_logging` resolved its level having already read `LOG_LEVEL`."""
        monkeypatch.setenv(LOG_LEVEL_ENV, 'DEBUG')
        monkeypatch.setenv(RESOLVED_LOG_LEVEL_ENV, str(logging.ERROR))

        configure_process_logging()

        assert logging.root.level == logging.ERROR

    def test_an_unconfigured_parent_leaves_this_process_at_info(self):
        """Nothing asked for a threshold. That must not read as a request for silence."""
        configure_process_logging()

        assert logging.root.level == logging.INFO

    @pytest.mark.parametrize('variable', [LOG_LEVEL_ENV, RESOLVED_LOG_LEVEL_ENV])
    def test_an_empty_value_is_refused(self, monkeypatch, variable):
        """An empty value is one somebody set, and `init_logging` raises on it. Reading it as INFO
        here would make one variable mean two things."""
        monkeypatch.setenv(variable, '')

        with pytest.raises(ValueError, match=variable):
            configure_process_logging()


class TestTheThresholdHoldsAgainstAComponentsOwnLevel:
    """A logger's own level admits a record; past that only a handler's threshold can stop it."""

    def test_a_component_that_set_its_own_level_does_not_escape_the_threshold(self, monkeypatch, capfd):
        monkeypatch.setenv(LOG_LEVEL_ENV, 'ERROR')
        configure_process_logging()

        with _level(A_SELF_LEVELLED_COMPONENT, logging.INFO):
            logging.getLogger(A_SELF_LEVELLED_COMPONENT).info(CHILD_LINE)

        assert CHILD_LINE not in capfd.readouterr().err

    def test_a_record_at_the_threshold_still_reaches_the_stream(self, monkeypatch, capfd):
        """A handler set stricter than the threshold would drop the run's own errors."""
        monkeypatch.setenv(LOG_LEVEL_ENV, 'ERROR')
        configure_process_logging()

        logging.getLogger(A_SELF_LEVELLED_COMPONENT).error(CHILD_LINE)

        assert CHILD_LINE in capfd.readouterr().err

    def test_the_handler_init_logging_installs_carries_the_threshold(self, capfd):
        """`coloredlogs.install` replaces the handler `configure_process_logging` had capped, so the
        main process keeps this ceiling only while the replacement carries the level too."""
        init_logging('ERROR')

        with _level(A_SELF_LEVELLED_COMPONENT, logging.INFO):
            logging.getLogger(A_SELF_LEVELLED_COMPONENT).info(CHILD_LINE)

        assert CHILD_LINE not in capfd.readouterr().err

    def test_the_main_process_still_passes_a_record_at_the_threshold(self, capfd):
        """A colour handler set stricter than the threshold would drop the run's own errors."""
        init_logging('ERROR')

        logging.getLogger(A_SELF_LEVELLED_COMPONENT).error(CHILD_LINE)

        assert CHILD_LINE in capfd.readouterr().err


class TestTheLibraryPins:
    """WARNING is a floor under a noisy library, not the level it is assigned."""

    def test_a_stricter_level_already_set_is_kept(self):
        """Lowering it would start a library the application had silenced, and only when the
        silencing happened to run before `init_logging`."""
        with _level(A_NOISY_LIBRARY, logging.ERROR):
            configure_process_logging()

            assert logging.getLogger(A_NOISY_LIBRARY).level == logging.ERROR

    def test_a_library_nobody_pinned_takes_the_floor(self):
        """Keeping a stricter level must not turn the floor off for NOTSET, which is where every
        pinned library starts."""
        with _level(A_NOISY_LIBRARY, logging.NOTSET):
            configure_process_logging()

            assert logging.getLogger(A_NOISY_LIBRARY).level == logging.WARNING

    def test_a_looser_level_is_raised_to_the_floor(self):
        with _level(A_NOISY_LIBRARY, logging.DEBUG):
            configure_process_logging()

            assert logging.getLogger(A_NOISY_LIBRARY).level == logging.WARNING

    def test_a_second_call_at_a_lower_level_lowers_the_pin(self):
        """A pin is this module's own output. Reading it back as a setting to preserve would make it
        the next call's input, the same trap `RESOLVED_LOG_LEVEL_ENV` keeps the threshold out of."""
        init_logging('ERROR')
        assert logging.getLogger(A_NOISY_LIBRARY).level == logging.ERROR

        init_logging('DEBUG')

        assert logging.getLogger(A_NOISY_LIBRARY).level == logging.WARNING

    def test_a_level_equal_to_the_pin_it_received_is_still_the_applications_own(self):
        """A pin and the application's own level coincide whenever the threshold is what the
        application asked for, so telling them apart by value alone would discard the setting."""
        with _level(A_NOISY_LIBRARY, logging.ERROR):
            init_logging('ERROR')
            init_logging('DEBUG')

            assert logging.getLogger(A_NOISY_LIBRARY).level == logging.ERROR

    def test_a_level_set_between_two_calls_is_kept(self):
        """Recomputing past this module's own pin must not discard a setting that arrived after the
        first call: the floor exists to be independent of that ordering."""
        init_logging('DEBUG')

        with _level(A_NOISY_LIBRARY, logging.ERROR):
            init_logging('DEBUG')

            assert logging.getLogger(A_NOISY_LIBRARY).level == logging.ERROR


class TestASecondCallCanStillChangeTheThreshold:
    """A resolved level is `init_logging`'s output, so it must not become its own input next call."""

    def test_the_second_call_wins(self):
        init_logging()
        init_logging('ERROR')

        assert logging.root.level == logging.ERROR
        assert LOG_LEVEL_ENV not in os.environ, f'{LOG_LEVEL_ENV} was written'

    def test_a_child_spawned_after_it_is_suppressed(self, capfd):
        init_logging()
        init_logging('ERROR')

        err = _stderr_of_a_logging_child(capfd)

        assert CHILD_LINE not in err, err
        assert STOP_LINE not in err, err

    def test_log_level_takes_precedence_over_the_argument(self, monkeypatch):
        monkeypatch.setenv(LOG_LEVEL_ENV, 'DEBUG')

        init_logging('ERROR')

        assert logging.root.level == logging.DEBUG


class TestTheMainProcessConfiguresItselfLikeAChild:
    """One function configures the main process and a subprocess alike, so a threshold means the
    same thing on either side of a spawn."""

    def test_a_noisy_library_is_pinned_in_the_main_process_too(self):
        """The main process runs the drivers a child runs, so the same libraries would be as loud in it."""
        init_logging('DEBUG')

        assert logging.getLogger(A_NOISY_LIBRARY).level == logging.WARNING

    def test_the_main_process_takes_the_level_it_publishes(self):
        init_logging('WARNING')

        assert logging.root.level == logging.WARNING
        assert os.environ[RESOLVED_LOG_LEVEL_ENV] == str(logging.WARNING)


class TestALevelTheChildCannotName:
    """The threshold reaches a child as a number, so it does not depend on the child's own registry."""

    def test_a_level_registered_only_in_the_parent_still_starts_the_child(self, capfd):
        # `addLevelName` is process-local and a spawn starts an empty registry, so this level
        # carried as a name would raise in the child and end the World before the loop ran.
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
        """Carrying numbers must not stop a name that is not a level being rejected."""
        with pytest.raises(ValueError, match='EROR'):
            init_logging('EROR')


class TestPerComponentLevelsCrossASpawn:
    """A spawn starts an empty registry, so a level set on a component reaches a child only if it
    is carried there."""

    def test_a_component_silenced_in_the_parent_is_silent_in_the_child(self, capfd):
        init_logging('INFO')

        with _level(A_COMPONENT, logging.ERROR):
            err = _stderr_of_a_logging_child(capfd, component_loop)

        assert CHILD_LINE not in err, err

    def test_a_component_nobody_set_a_level_on_still_logs_in_the_child(self, capfd):
        """Carrying the levels a parent has must not silence one it never set."""
        init_logging('INFO')

        err = _stderr_of_a_logging_child(capfd, component_loop)

        assert CHILD_LINE in err, err


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

        # A threshold that reached past `pimm.utils` would take this line with it.
        assert _world_lines(caplog) == [WORLD_LINE], caplog.text

    def test_both_components_log_at_a_shared_threshold(self, caplog):
        """With no level set both lines land, so a test that sets one and sees silence is reading
        the level rather than a component that never logged."""
        caplog.set_level(logging.INFO)

        _emit_one_info_line_from_each_component()

        assert len(_counter_lines(caplog)) == 1, caplog.text
        assert _world_lines(caplog) == [WORLD_LINE], caplog.text
