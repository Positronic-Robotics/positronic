import json
import os
import stat
from pathlib import Path

import pytest

import pimm
from positronic import keys
from positronic.cli.eval import run_state


def read_state(path: Path) -> dict:
    """The file as a reader in another repository sees it: bytes, parsed as JSON, nothing else."""
    return json.loads(path.read_text())


@pytest.fixture
def named_run(tmp_path, monkeypatch):
    """A run that is being watched: an id and a directory of its own. Yields its state path."""
    monkeypatch.setenv(run_state.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(tmp_path))
    return run_state.state_path('batch-1')


# A round, in the virtual time these tests run under. Positive because pimm reserves zero for
# `Yield`; the clock is virtual, so nothing waits for it.
TICK_S = 0.001


class Camera(pimm.ControlSystem):
    """A producer that emits `count` frames and stops, standing in for a camera that comes up."""

    def __init__(self, count: int = 1):
        self.frame = pimm.ControlSystemEmitter(self)
        self._count = count

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        emitted = 0
        while not should_stop.value:
            if emitted < self._count:
                self.frame.emit(emitted, clock.now_ns())
                emitted += 1
            yield pimm.Sleep(TICK_S)


class CountingReceiver(pimm.ControlSystemReceiver):
    """A receiver that records how often it was read, to pin what the watch does NOT do."""

    def __init__(self, owner: pimm.ControlSystem):
        super().__init__(owner)
        self.reads = 0

    def read(self):
        self.reads += 1
        return super().read()


class Stopper(pimm.ControlSystem):
    """Ends the world after `rounds`, since the readiness watch never returns."""

    def __init__(self, rounds: int):
        self._rounds = rounds

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for _ in range(self._rounds):
            yield pimm.Sleep(TICK_S)


# --- the contract ---------------------------------------------------------------------------


def test_a_named_run_reports_that_it_has_started(named_run):
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    run_state.from_env()
    assert read_state(named_run) == {
        'run_id': 'batch-1',
        'phase': 'starting',
        'cameras_open': 0,
        'cameras_expected': 0,
        'console_port': None,
    }


def test_an_unnamed_run_reports_nothing(tmp_path, monkeypatch):
    """Every ordinary invocation: nothing named it, so it writes no file anywhere."""
    monkeypatch.delenv(run_state.RUN_ID_ENV, raising=False)
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(tmp_path))
    state = run_state.from_env()
    assert not state.enabled
    state.report(run_state.Phase.WARMING_UP)
    assert list(tmp_path.iterdir()) == []


def test_the_path_is_scoped_to_the_run(tmp_path, monkeypatch):
    """Two runs never share a file, which is what makes a leftover from a previous one inert."""
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(tmp_path))
    assert run_state.state_path('a') != run_state.state_path('b')
    assert run_state.state_path('a').parent == tmp_path


def test_a_run_id_that_is_not_one_path_segment_reports_nowhere(tmp_path, monkeypatch):
    """It would otherwise write into a directory nobody agreed on."""
    monkeypatch.setenv(run_state.RUN_ID_ENV, '../elsewhere')
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(tmp_path))
    assert not run_state.from_env().enabled
    assert list(tmp_path.iterdir()) == []


def test_a_relative_directory_reports_nowhere(monkeypatch):
    """A relative path resolves per account, so the two would address different files."""
    monkeypatch.setenv(run_state.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(run_state.STATE_DIR_ENV, 'run/lock')
    assert not run_state.from_env().enabled


def test_the_file_is_readable_by_another_account(named_run):
    """`mkstemp` creates 0600 whatever the umask, and the reader is not this account."""
    run_state.from_env()
    assert stat.S_IMODE(named_run.stat().st_mode) == 0o644


def test_a_report_leaves_no_temporary_file_behind(named_run):
    state = run_state.from_env()
    state.report(run_state.Phase.WARMING_UP)
    assert [p.name for p in named_run.parent.iterdir()] == [named_run.name]


def test_an_unwritable_directory_does_not_stop_the_run(tmp_path, monkeypatch, caplog):
    """The arm is mid-episode; there is no version of "the progress report failed" worth raising."""
    # rules-allow: hardcoded-keys — a fragment of this module's own log line, not a name any
    # other scope agrees on.
    if os.geteuid() == 0:
        pytest.skip('root writes into a mode-500 directory, so the refused branch cannot be reached')
    closed = tmp_path / 'closed'
    closed.mkdir(mode=0o500)
    monkeypatch.setenv(run_state.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(closed))
    state = run_state.from_env()
    state.report(run_state.Phase.WARMING_UP)
    assert state.state.phase is run_state.Phase.WARMING_UP
    assert 'could not be written' in caplog.text


def test_a_persistent_write_failure_is_logged_once(tmp_path, monkeypatch, caplog):
    """A directory that cannot be written is one log line, not one per transition."""
    if os.geteuid() == 0:
        pytest.skip('root writes into a mode-500 directory, so the refused branch cannot be reached')
    closed = tmp_path / 'closed'
    closed.mkdir(mode=0o500)
    monkeypatch.setenv(run_state.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(run_state.STATE_DIR_ENV, str(closed))
    state = run_state.from_env()
    for phase in (run_state.Phase.WARMING_UP, run_state.Phase.WORLD_UP, run_state.Phase.READY):
        state.report(phase)
    assert caplog.text.count('could not be written') == 1


def test_a_report_replaces_the_file_rather_than_truncating_it(named_run):
    """A poll must never meet a half-written file: the visible file is only ever swapped."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    run_state.from_env()
    before = named_run.stat().st_ino
    state = run_state.StateFile(named_run, 'batch-1')
    state.report(run_state.Phase.READY, cameras_open=2, cameras_expected=2)
    assert named_run.stat().st_ino != before
    assert read_state(named_run)['phase'] == 'ready'


def test_a_report_carries_unnamed_fields_forward(named_run):
    """A camera count is measured once and holds, so a phase change must not reset it."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    state.report(run_state.Phase.WORLD_UP, cameras_expected=2, console_port=8080)
    state.report(run_state.Phase.READY, cameras_open=2)
    assert read_state(named_run) == {
        'run_id': 'batch-1',
        'phase': 'ready',
        'cameras_open': 2,
        'cameras_expected': 2,
        'console_port': 8080,
    }


# --- the transitions ------------------------------------------------------------------------


def _drive(state: run_state.StateFile, cameras: dict[str, Camera], rounds: int = 6) -> run_state.Readiness:
    """Run a world holding `cameras` and a readiness watch over them, for a bounded number of rounds."""
    readiness = run_state.Readiness(state, list(cameras), console_port=8080, poll_interval_s=TICK_S)
    with pimm.World(virtual_time=True) as world:
        for name, camera in cameras.items():
            world.connect(camera.frame, readiness.cameras[name])
        world.run([*cameras.values(), readiness, Stopper(rounds)])
    return readiness


def test_the_world_coming_up_is_reported_before_any_camera(named_run):
    """Reported on its own, ahead of any camera — the distinction between a UI that exists and
    one that works."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    _drive(state, {keys.WRIST_IMAGE: Camera(count=0)}, rounds=1)
    assert read_state(named_run) == {
        'run_id': 'batch-1',
        'phase': 'world_up',
        'cameras_open': 0,
        'cameras_expected': 1,
        'console_port': 8080,
    }


def test_every_camera_delivering_a_frame_is_readiness(named_run):
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    _drive(state, {keys.WRIST_IMAGE: Camera(), keys.EXTERIOR_IMAGE: Camera()})
    got = read_state(named_run)
    assert got['phase'] == 'ready'
    assert (got['cameras_open'], got['cameras_expected']) == (2, 2)


def test_a_camera_that_never_opens_holds_readiness_back(named_run):
    """One camera up and its neighbour missing tears the World down as thoroughly as both."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    _drive(state, {keys.WRIST_IMAGE: Camera(), keys.EXTERIOR_IMAGE: Camera(count=0)})
    got = read_state(named_run)
    assert got['phase'] == 'world_up'
    assert (got['cameras_open'], got['cameras_expected']) == (1, 2)


def test_a_run_with_no_cameras_is_ready_at_once(named_run):
    """A rig with nothing to open has no device to wait on — the absence of a question."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    _drive(state, {})
    assert read_state(named_run)['phase'] == 'ready'


def test_a_camera_is_read_only_until_it_is_seen(named_run):
    """A read takes the lock four systems contend for and copies the whole frame out."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    camera = Camera(count=4)
    readiness = run_state.Readiness(state, [keys.WRIST_IMAGE], poll_interval_s=TICK_S)
    counting = CountingReceiver(readiness)
    readiness.cameras[keys.WRIST_IMAGE] = counting
    with pimm.World(virtual_time=True) as world:
        world.connect(camera.frame, counting)
        world.run([camera, readiness, Stopper(8)])
    assert read_state(named_run)['phase'] == 'ready'
    # Once, on the round that found it. Every later round skips a channel already known to be up.
    assert counting.reads == 1


def test_the_watch_does_not_end_the_world(named_run):
    """A control system that returns sets the world's stop event, which would end the run the
    moment every camera came up."""
    # rules-allow: hardcoded-keys — the wire shape, asserted the way a reader in another
    # repository sees it. Written through the constants these tests would still pass if one
    # were renamed, which is the single thing they exist to catch.
    state = run_state.from_env()
    camera = Camera()
    stopper = Stopper(20)
    readiness = run_state.Readiness(state, [keys.WRIST_IMAGE], poll_interval_s=TICK_S)
    rounds = 0

    def counted_rounds(should_stop, clock, inner=stopper.run):
        nonlocal rounds
        for command in inner(should_stop, clock):
            rounds += 1
            yield command

    stopper.run = counted_rounds
    with pimm.World(virtual_time=True) as world:
        world.connect(camera.frame, readiness.cameras[keys.WRIST_IMAGE])
        world.run([camera, readiness, stopper])
    assert rounds == 20
    assert read_state(named_run)['phase'] == 'ready'
