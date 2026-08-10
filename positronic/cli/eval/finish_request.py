"""An orchestrator ends an attended run by writing a file; the run polls it and stops after the
current episode, through the ordinary shutdown — a signal unwinds nothing (recording unclosed,
dataset not uploaded, arm keeps its token).

The contract, for a writer in another repository:

- path: `FINISH_REQUEST_PREFIX` + `run_id`, inside `DEFAULT_FINISH_REQUEST_DIR` (overridden by
  `FINISH_REQUEST_DIR_ENV`; absolute); `run_id` is one path segment. Per run, so a leftover is inert.
- content `{"action": "finish", "run_id": "<id>"}`; further keys ignored.
- the writer creates it world-readable (a `077` umask leaves it unreadable) and never unlinks it;
  the run only reads it, and only where `run_id` matches the run's own `RUN_ID_ENV`; unset, no
  poller is installed.
- intent is monotonic: never withdrawn, so a writer unsure its write landed re-asserts.
- the only acknowledgement is `ACK_LOG_MARKER` in the run's log.
- an absent file is the ordinary state: no request. Any other failure to read the file as a request
  addressed to this run raises and ends the World — the system is ours end to end, so a contract
  violation is breakage to surface, not input to tolerate.
"""

import json
import logging
import os
import stat
from enum import StrEnum
from pathlib import Path
from typing import Any

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

# A request is around 120 bytes; a file past this bound is not one.
MAX_REQUEST_BYTES = 64 * 1024


def request_dir() -> Path:
    """Where requests are read from — the env override converted here, where the string enters."""
    override = os.environ.get(FINISH_REQUEST_DIR_ENV)
    return Path(override) if override else DEFAULT_FINISH_REQUEST_DIR


def request_path(this_run: str) -> Path:
    """Where a request addressed to `this_run` is read from — the whole of the path contract."""
    return request_dir() / f'{FINISH_REQUEST_PREFIX}{this_run}'


def evaluate(path: Path, this_run: str) -> bool:
    """Whether a finish request addressed to `this_run` is waiting at `path`.

    False means there is no file, which is every run nobody has asked. Anything else at that path
    raises: the writer and the run are one system, so a file that is not the request the contract
    describes is breakage to surface rather than input to tolerate.
    """
    blob = _read_bytes(path)
    if blob is None:
        return False
    request = json.loads(blob.decode())
    if not isinstance(request, dict):
        raise ValueError(f'finish request at {path} is {type(request).__name__}, not an object')
    _assert_addressed_finish(path, request, this_run)
    return True


def _read_bytes(path: Path) -> bytes | None:
    """The file's bytes, or `None` where there is no file."""
    # O_NONBLOCK: a FIFO here would hang the foreground poller. O_NOFOLLOW: a symlink fails (ELOOP).
    # fstat reads the open descriptor, not the path, so nothing swaps under the check.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f'finish request at {path} is not a regular file ({stat.filemode(mode)})')
        with os.fdopen(fd, 'rb', closefd=False) as f:
            # One byte past the bound, so an oversized file is caught rather than truncated into
            # something that might parse.
            blob = f.read(MAX_REQUEST_BYTES + 1)
    finally:
        os.close(fd)
    if len(blob) > MAX_REQUEST_BYTES:
        raise ValueError(f'finish request at {path} is larger than {MAX_REQUEST_BYTES} bytes')
    return blob


def _assert_addressed_finish(path: Path, request: dict[str, Any], this_run: str) -> None:
    """Raise unless the request object asks `this_run` to finish."""
    named = request.get(ACTION_KEY)
    # Checked before the membership test: JSON's containers are unhashable, so one reaching it raises
    # `TypeError` instead of the `ValueError` that names what is wrong.
    if not isinstance(named, str):
        raise ValueError(f'finish request at {path} names action {named!r}, which is not a string')
    if named not in frozenset(Action):
        raise ValueError(f'finish request at {path} names action {named!r}, which this run does not implement')
    # A `StrEnum` member compares equal to its own string, so a bare comparison would leave every
    # later reader holding a `str` the contract calls an `Action`.
    action = Action(named)
    if action is not Action.FINISH:
        raise ValueError(f'finish request at {path} names action {action.value!r}, not {Action.FINISH.value!r}')
    addressee = request.get(RUN_ID_KEY)
    if addressee != this_run:
        raise ValueError(f'finish request at {path} names run {addressee!r}, not {this_run!r}')


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
        # The grant, latched: once made it is never withdrawn, and it outlives the World that read it.
        self._granted = False

    @property
    def granted(self) -> bool:
        """Whether a request addressed to this run has been granted."""
        return self._granted

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        # The yielded `Sleep` is the whole pace, so the interval is the world's clock — the one the
        # episodes run on. A wall-clock pace would let whole simulated episodes pass between reads.
        while not should_stop.value:
            if not self._granted:
                self._granted = evaluate(self._path, self._run)
                if self._granted:
                    logger.info('%s: run %s stops after the current episode', ACK_LOG_MARKER, self._run)
            if self._granted:
                # Every round, not once: the harness reads a signal, and a single emit that landed
                # before it bound would be a request that silently never arrives.
                self.requested.emit(True, clock.now_ns())
            yield pimm.Sleep(self._poll_interval_s)


def names_one_segment(this_run: str) -> bool:
    """Whether `this_run` can be a filename, which is what scoping the path by run id requires.

    A run id carrying a separator would address a file in a directory nobody agreed on.
    """
    return bool(this_run) and this_run not in ('.', '..') and '/' not in this_run and '\0' not in this_run


def from_env() -> FinishRequest | None:
    """The control system this run should carry, or None where nothing named the run."""
    this_run = os.environ.get(RUN_ID_ENV)
    if not this_run:
        return None
    # Raised at launch, before an episode runs: a run that cannot be addressed is a misconfigured
    # deploy, and discovering it is better than running unpollable for hours.
    if not names_one_segment(this_run):
        raise ValueError(f'{RUN_ID_ENV}={this_run!r} is not a single path segment, so no request can address this run')
    directory = request_dir()
    if not directory.is_absolute():
        raise ValueError(
            f'{FINISH_REQUEST_DIR_ENV}={directory} is not absolute, so the writer and this run would not '
            'resolve it to the same directory'
        )
    path = request_path(this_run)
    logger.info('run %s will stop after the current episode if %s asks it to', this_run, path)
    return FinishRequest(path, this_run)
