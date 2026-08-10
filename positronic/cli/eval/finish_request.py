"""Ending an attended run from outside it, without a signal.

A run driven by the local eval UI serves no operator surface, so an orchestrator has nowhere to post
`finish`. It writes a small file instead; the run polls it and, once the episode in progress has
completed, stops the way a plan running out stops. A signal is not an alternative: nothing unwinds
the World, so the recording is not closed, the dataset is not uploaded, and the arm keeps its token.

THE CONTRACT, which a writer in another repository implements against:

  path     `<dir>/positronic_rollout_finish.<run_id>`; `<dir>` is `/run/lock` unless
           `ROLLOUT_FINISH_REQUEST_DIR` names another. Absolute — writer and run are different
           accounts, each resolving a relative path against its own directory. Per run, so a
           leftover is inert. `run_id` is one path segment: no separator, not `.` or `..`.
  content  one JSON object, `{"action": "finish", "run_id": "<id>"}`; further keys are ignored.
  writer   creates it world-readable (a `077` umask leaves it unreadable by the run) and never
           unlinks it; the run only ever reads it, and only where `run_id` equals its own
           `ROLLOUT_RUN_ID`.
  intent   monotonic — never withdrawn, so a writer unsure its write landed re-asserts it.
  ack      `ACK_LOG_MARKER` in the run's log, and nothing else.

It FAILS CLOSED, where closed means the run keeps running: anything that is not a readable request
addressed to this run is ignored and logged once. Inert when `ROLLOUT_RUN_ID` is unset.
"""

import json
import logging
import os
import stat
from enum import StrEnum
from pathlib import Path

import pimm

logger = logging.getLogger(__name__)

# Where requests are read from, and the name each takes there. Overridable so a test, or a rig
# running more than one simulated run, gets a directory of its own.
FINISH_REQUEST_DIR_ENV = 'ROLLOUT_FINISH_REQUEST_DIR'
DEFAULT_FINISH_REQUEST_DIR = Path('/run/lock')
FINISH_REQUEST_PREFIX = 'positronic_rollout_finish.'
# The run's own identity, set by whatever launched it. Its absence is what makes this inert.
RUN_ID_ENV = 'ROLLOUT_RUN_ID'

# The object's two required fields, and the closed set of actions it may name.
ACTION_KEY = 'action'
RUN_ID_KEY = 'run_id'


class Action(StrEnum):
    FINISH = 'finish'


# How often the file is read; the wait this adds to a finish is bounded by it.
POLL_INTERVAL_S = 2.0

# Logged the moment a request is granted, and the writer's only acknowledgement: it separates a
# request still being worked through — the run is mid-episode, which takes minutes — from one that
# never arrived. From outside the process both look like a run that is still running.
ACK_LOG_MARKER = 'rollout finish request granted'

# Any account on the rig may create this file, so the read is bounded rather than trusted; a real
# request is around 120 bytes.
MAX_REQUEST_BYTES = 64 * 1024

# EVERY way a file can fail to become a request. `ValueError` covers the decode and the parse, both
# subclasses of it; `RecursionError` is the JSON parser's nesting limit, which is not one.
UNREADABLE = (OSError, ValueError, RecursionError)


def request_dir() -> Path:
    """Where requests are read from — the env override converted here, where the string enters."""
    override = os.environ.get(FINISH_REQUEST_DIR_ENV)
    return Path(override) if override else DEFAULT_FINISH_REQUEST_DIR


def request_path(this_run: str) -> Path:
    """Where a request addressed to `this_run` is read from — the whole of the path contract."""
    return request_dir() / f'{FINISH_REQUEST_PREFIX}{this_run}'


def evaluate(path: Path, this_run: str) -> tuple[bool, str]:
    """Whether `path` holds a finish request addressed to `this_run`, and why not when it does not.

    Every negative is a reason to keep running, so the caller acts on the bool alone. The reason is
    returned rather than logged so the caller can report a persistent one once instead of on every
    poll. TOTAL: any failure to read the file as a request addressed to this run is a refusal, since
    an exception here would end the World mid-episode.
    """
    try:
        return _read(path, this_run)
    except FileNotFoundError:
        # The ordinary state of a run nobody has asked, so not a fault worth a reason.
        return False, ''
    except UNREADABLE as e:
        return False, f'finish request at {path} could not be read as a request ({e!r}); continuing to run'


def _read(path: Path, this_run: str) -> tuple[bool, str]:
    """The staged read: a refusal is returned, any other failure raises, and all of those are in `UNREADABLE`."""
    # Any account on the rig may create this path, so what it names need not be a file. `O_NONBLOCK`
    # stops a FIFO holding the open until somebody writes — this poller runs in the foreground, so
    # that wait would hang the whole run. `O_NOFOLLOW` refuses a symlink with `ELOOP`. The `fstat`
    # answers for a directory or a device, and reads the open descriptor, not the path, so nothing
    # can be swapped underneath the check.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            return False, f'finish request at {path} is not a regular file ({stat.filemode(mode)}); continuing to run'
        with os.fdopen(fd, 'rb', closefd=False) as f:
            # One byte past the bound, so an oversized file is recognised rather than truncated into
            # something that might parse.
            blob = f.read(MAX_REQUEST_BYTES + 1)
    finally:
        os.close(fd)
    if len(blob) > MAX_REQUEST_BYTES:
        return False, f'finish request at {path} is larger than {MAX_REQUEST_BYTES} bytes; continuing to run'
    request = json.loads(blob.decode())
    if not isinstance(request, dict):
        return False, f'finish request at {path} is {type(request).__name__}, not an object; continuing to run'
    named = request.get(ACTION_KEY)
    # The type is checked before the value because JSON's containers are unhashable: an array or
    # object reaching the membership test below raises `TypeError`, which is not in `UNREADABLE`.
    # An absent key arrives here as `None` and is the same fact — nothing an action reads from.
    if not isinstance(named, str):
        return (
            False,
            f'finish request at {path} names action {named!r}, which is not a string this run can read; '
            'continuing to run',
        )
    if named not in frozenset(Action):
        return (
            False,
            f'finish request at {path} names action {named!r}, which this run does not implement; continuing to run',
        )
    # A `StrEnum` member compares equal to its own string, so a bare comparison would leave every
    # later reader holding a `str` the contract calls an `Action`.
    action = Action(named)
    if action is not Action.FINISH:
        return (
            False,
            f'finish request at {path} names action {action.value!r}, not {Action.FINISH.value!r}; continuing to run',
        )
    addressee = request.get(RUN_ID_KEY)
    if addressee != this_run:
        # Routine rather than a fault: a previous run's request outliving it.
        return False, f'finish request at {path} names run {addressee!r}, not {this_run!r}; continuing to run'
    return True, ''


class FinishRequest(pimm.ControlSystem):
    """Emits True once a finish request addressed to this run appears, and keeps emitting it.

    It never returns from its loop: a control system that returns sets the world's stop event, which
    would end the run wherever it happened to be — the mid-episode stop this exists to replace. The
    harness owns when it is safe to stop; this only reports that someone asked.
    """

    def __init__(self, path: Path, this_run: str, poll_interval_s: float = POLL_INTERVAL_S):
        self.requested = pimm.ControlSystemEmitter[bool](self)
        self._path = path
        self._run = this_run
        self._poll_interval_s = poll_interval_s
        # The last refusal reported, so a request that stays wrong is logged once rather than every
        # poll. Reported again when the reason changes, which is what a replaced file looks like.
        self._reported = ''
        # One object serves every World of a sweep, so the grant has to outlive any single World.
        self._granted = False

    @property
    def granted(self) -> bool:
        """Whether a request addressed to this run has been granted."""
        return self._granted

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        # The yielded `Sleep` is the whole pace, so the interval is measured on the world's clock —
        # the one the run's episodes are measured on. A wall-clock pace is incommensurable with a
        # simulated sweep, which advances episodes as fast as the machine allows: entire episodes
        # pass between two reads, and a request written during one is still unread when the harness
        # decides whether to open the next.
        while not should_stop.value:
            if not self._granted:
                self._granted, reason = evaluate(self._path, self._run)
                if self._granted:
                    logger.info('%s: run %s stops after the current episode', ACK_LOG_MARKER, self._run)
                elif reason != self._reported:
                    self._reported = reason
                    if reason:
                        logger.error(reason)
            if self._granted:
                # Every round, not once: the harness reads a signal, and a single emit that landed
                # before it bound would be a request that silently never arrives.
                self.requested.emit(True, clock.now_ns())
            yield pimm.Sleep(self._poll_interval_s)


def names_one_segment(this_run: str) -> bool:
    """Whether `this_run` can be a filename, which is what scoping the path by run id requires.

    A run id carrying a separator would address a file in a directory nobody agreed on, so a run
    named that way is left with no poller rather than polling somewhere unintended.
    """
    return bool(this_run) and this_run not in ('.', '..') and '/' not in this_run and '\0' not in this_run


def from_env() -> FinishRequest | None:
    """The control system this run should carry, or None where nothing named the run."""
    this_run = os.environ.get(RUN_ID_ENV)
    if not this_run:
        return None
    if not names_one_segment(this_run):
        logger.error(
            '%s=%r is not a single path segment, so no finish request can address this run', RUN_ID_ENV, this_run
        )
        return None
    directory = request_dir()
    if not directory.is_absolute():
        # Each account resolves a relative path against its own working directory, so the writer and
        # this run would address different files from the same configuration.
        logger.error(
            '%s=%s is not absolute, so the writer and this run would not resolve it to the same '
            'directory; no finish request can address this run',
            FINISH_REQUEST_DIR_ENV,
            directory,
        )
        return None
    path = request_path(this_run)
    logger.info('run %s will stop after the current episode if %s asks it to', this_run, path)
    return FinishRequest(path, this_run)
