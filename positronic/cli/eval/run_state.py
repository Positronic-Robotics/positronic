"""Reporting a run's own progress to whatever launched it.

From outside, "the run exists" and "the operator has a screen" are minutes apart and look
identical. This writes down which of the two it is.

THE CONTRACT, which a reader in another repository implements against:

  path     `<dir>/positronic_rollout_state.<run_id>`; `<dir>` is `/run/lock` unless
           `ROLLOUT_RUN_STATE_DIR` names another. Absolute, since writer and reader are different
           accounts. `run_id` is one path segment: no separator, not `.` or `..`.
  content  one JSON object, `{"run_id", "phase", "cameras_open", "cameras_expected",
           "console_port"}`. A reader ignores keys it does not know.
  writer   this run alone, mode 0644 explicitly (a `077` umask would leave it unreadable by the
           reader), replaced atomically, so a poll never sees half a write.
  phase    one of `Phase`, the run NOW rather than the furthest it has been. A sweep raising a
           second World reports that World coming up.
  console  the PORT, or null where the run has no console. Not a URL: the run binds `0.0.0.0` and
           cannot know which address a reader comes from, so the host is the reader's to supply.
  absent   NOT a run that failed — a run predating this module. A reader reports unknown.

Best-effort: a failed write is logged, never raised into the run, which is mid-episode. Inert when
`ROLLOUT_RUN_ID` is unset, which is every ordinary invocation.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import pimm

logger = logging.getLogger(__name__)

# Overridable so a test, or a rig running more than one simulated run, gets its own directory.
STATE_DIR_ENV = 'ROLLOUT_RUN_STATE_DIR'
DEFAULT_STATE_DIR = Path('/run/lock')
STATE_PREFIX = 'positronic_rollout_state.'
# The run's own identity, set by whatever launched it. Its absence is what makes this inert.
RUN_ID_ENV = 'ROLLOUT_RUN_ID'

# The object's fields.
RUN_ID_KEY = 'run_id'
PHASE_KEY = 'phase'
CAMERAS_OPEN_KEY = 'cameras_open'
CAMERAS_EXPECTED_KEY = 'cameras_expected'
CONSOLE_PORT_KEY = 'console_port'


class Phase(StrEnum):
    """What the run is doing, in the order it does it."""

    # The process is up and holds this contract; separates it from a writer too old to have one.
    STARTING = 'starting'
    # Driving the policy endpoints through their cold start: seconds to a quarter of an hour.
    WARMING_UP = 'warming_up'
    # The World is up and its background processes, the operator's UI among them, are spawned.
    # NOT usable yet: the cameras open after this, and one that will not takes the run down with it.
    WORLD_UP = 'world_up'
    # Every camera the run declared has delivered a frame.
    READY = 'ready'


# How often the cameras are counted; each read is a shared-memory flag, so this paces nothing.
POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class State:
    """One report: everything the file says, at one moment."""

    run_id: str
    phase: Phase
    cameras_open: int = 0
    cameras_expected: int = 0
    console_port: int | None = None

    def as_json(self) -> str:
        return json.dumps({
            RUN_ID_KEY: self.run_id,
            PHASE_KEY: self.phase.value,
            CAMERAS_OPEN_KEY: self.cameras_open,
            CAMERAS_EXPECTED_KEY: self.cameras_expected,
            CONSOLE_PORT_KEY: self.console_port,
        })


def state_dir() -> Path:
    """Where state is written — the env override converted here, where the string enters."""
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else DEFAULT_STATE_DIR


def state_path(this_run: str) -> Path:
    """Where `this_run` reports — the whole of the path contract."""
    return state_dir() / f'{STATE_PREFIX}{this_run}'


def names_one_segment(this_run: str) -> bool:
    """Whether `this_run` can be a filename, which is what scoping the path by run id requires.

    One carrying a separator would write to a directory nobody agreed on, so it reports nowhere.
    """
    return bool(this_run) and this_run not in ('.', '..') and '/' not in this_run and '\0' not in this_run


class StateFile:
    """The run's state, and the file it is reported through.

    An instance with no path is INERT — it holds the state and writes nothing — which is what every
    ordinary invocation gets, and why no call site guards.
    """

    def __init__(self, path: Path | None, this_run: str):
        self._path = path
        self._state = State(run_id=this_run, phase=Phase.STARTING)
        # The last failure reported, so an unwritable directory is one log line, not one a
        # transition. Reported again when the reason changes.
        self._reported = ''

    @property
    def enabled(self) -> bool:
        """Whether anything is being reported. False on an ordinary run."""
        return self._path is not None

    @property
    def state(self) -> State:
        return self._state

    def report(self, phase: Phase, **fields) -> None:
        """Record `phase`, carrying every field not named forward, and write the result.

        A camera count is measured once and holds, so a phase change must not reset it.
        """
        self._state = replace(self._state, phase=phase, **fields)
        self._write()

    def _write(self) -> None:
        """Replace the file with the current state, atomically. Never raises into the run."""
        if self._path is None:
            return
        blob = self._state.as_json()
        try:
            # The destination's own directory, so the rename is within one filesystem: atomic.
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f'{self._path.name}.')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(blob)
                # mkstemp creates 0600 whatever the umask, and the reader is another account.
                os.chmod(tmp, 0o644)
                os.replace(tmp, self._path)
            except OSError:
                os.unlink(tmp)
                raise
        except OSError as e:
            reason = f'run state could not be written to {self._path} ({e!r}); continuing to run'
            if reason != self._reported:
                self._reported = reason
                logger.error(reason)
            return
        self._reported = ''


class Readiness(pimm.ControlSystem):
    """Reports the World coming up, and each of its cameras delivering its first frame.

    FOREGROUND only: the World spawns every background process before scheduling the first
    foreground round, which is what makes that round's World-up report true. It also never returns
    from its loop — a control system that returns sets the world's stop event.
    """

    def __init__(
        self,
        state: StateFile,
        camera_names: list[str],
        console_port: int | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
    ):
        self.cameras = pimm.ReceiverDict(self)
        # Touched here so the receivers exist to be connected and the count is fixed up front.
        for name in camera_names:
            _ = self.cameras[name]
        self._state = state
        self._console_port = console_port
        self._poll_interval_s = poll_interval_s
        self._open: set[str] = set()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        self._state.report(
            self._phase(), cameras_open=0, cameras_expected=len(self.cameras), console_port=self._console_port
        )
        while not should_stop.value:
            self._count()
            yield pimm.Sleep(self._poll_interval_s)

    def _phase(self) -> Phase:
        """Readiness, decided in one place: every camera the run declared has delivered a frame.

        A World with no cameras satisfies it the moment it is up — nothing to wait on.
        """
        return Phase.READY if len(self._open) == len(self.cameras) else Phase.WORLD_UP

    def _count(self) -> None:
        """Report any camera seen for the first time, and readiness once none is left.

        A receiver reads `None` until its first message ever arrives, so that transition IS
        first-frame. Only channels not yet seen are read: a read takes the lock the producer, the
        harness, the recorder and the UI contend for, and copies the whole frame out.
        """
        pending = [name for name in self.cameras if name not in self._open]
        opened = {name for name in pending if self.cameras[name].read() is not None}
        if not opened:
            return
        self._open |= opened
        self._state.report(self._phase(), cameras_open=len(self._open))


def from_env() -> StateFile:
    """The state file this run reports through — inert unless something named the run.

    Returns a file either way, so no caller asks whether anyone is listening. The first report is
    made here, so the file existing at all says the process speaks this contract.
    """
    this_run = os.environ.get(RUN_ID_ENV)
    if not this_run:
        return StateFile(None, '')
    if not names_one_segment(this_run):
        logger.error('%s=%r is not a single path segment, so this run cannot report its state', RUN_ID_ENV, this_run)
        return StateFile(None, this_run)
    directory = state_dir()
    if not directory.is_absolute():
        # A relative path resolves per account, so run and reader would address different files.
        logger.error(
            '%s=%s is not absolute, so this run and its reader would not resolve it to the same '
            'directory; this run cannot report its state',
            STATE_DIR_ENV,
            directory,
        )
        return StateFile(None, this_run)
    path = state_path(this_run)
    logger.info('run %s reports its progress to %s', this_run, path)
    state = StateFile(path, this_run)
    state.report(Phase.STARTING)
    return state
