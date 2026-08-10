"""An orchestrator ends an attended run by writing a file; the run polls it and stops after the
current episode, through the ordinary shutdown — a signal unwinds nothing (recording unclosed,
dataset not uploaded, arm keeps its token).

The contract, for a writer in another repository:

- path: `FINISH_REQUEST_PREFIX` + `run_id`, inside `DEFAULT_FINISH_REQUEST_DIR` (overridden by
  `FINISH_REQUEST_DIR_ENV`; absolute); `run_id` is one path segment. Per run, so a leftover is inert.
- content `{"action": "finish", "run_id": "<id>"}`; further keys ignored.
- the writer creates it world-readable (a `077` umask silently defeats this) and never unlinks it;
  the run only reads it, and only where `run_id` matches the run's own `RUN_ID_ENV`; unset, no
  poller is installed.
- intent is monotonic: never withdrawn, so a writer unsure its write landed re-asserts.
- the only acknowledgement is `ACK_LOG_MARKER` in the run's log.
- a file that cannot be read as a request — absent, unreadable, malformed — is refused: logged once,
  the run keeps going. Past the parse the checks are this run's own code, so a failure there raises.
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

# Any account on the rig may create this file, so the read is bounded rather than trusted; a real
# request is around 120 bytes.
MAX_REQUEST_BYTES = 64 * 1024

# Reading the file's bytes: an account this run does not control wrote them, so text that is not
# UTF-8, text that is not JSON, and nesting past the parser's limit are all ordinary.
# `UnicodeDecodeError` and `JSONDecodeError` are both `ValueError`; `RecursionError` is not.
MALFORMED_CONTENT = (ValueError, RecursionError)


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
    poll.

    The two halves are separated deliberately. Reaching and reading the file is external, and every
    way it fails is a refusal. What follows the parse is this run's own code over a `dict`, so a
    failure there can only be a defect and is left to raise.
    """
    blob, reason = _read_bytes(path)
    if blob is None:
        return False, reason
    request, reason = _parse(path, blob)
    if request is None:
        return False, reason
    return _asks_this_run_to_finish(path, request, this_run)


def _read_bytes(path: Path) -> tuple[bytes | None, str]:
    """The file's bytes, or `None` and the reason there are none to read."""
    # Any account on the rig may create this path, so what it names need not be a file. `O_NONBLOCK`
    # stops a FIFO holding the open until somebody writes — this poller runs in the foreground, so
    # that wait would hang the whole run. `O_NOFOLLOW` refuses a symlink with `ELOOP`. The `fstat`
    # answers for a directory or a device, and reads the open descriptor, not the path, so nothing
    # can be swapped underneath the check.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                return (
                    None,
                    f'finish request at {path} is not a regular file ({stat.filemode(mode)}); continuing to run',
                )
            with os.fdopen(fd, 'rb', closefd=False) as f:
                # One byte past the bound, so an oversized file is recognised rather than truncated
                # into something that might parse.
                blob = f.read(MAX_REQUEST_BYTES + 1)
        finally:
            os.close(fd)
    except FileNotFoundError:
        # The ordinary state of a run nobody has asked, so not a fault worth a reason.
        return None, ''
    except OSError as e:
        # What valid operation produces at a path any account on the rig may create: a permission, an
        # I/O error, the `ELOOP` a symlink takes under `O_NOFOLLOW`.
        return None, f'finish request at {path} could not be read as a request ({e!r}); continuing to run'
    if len(blob) > MAX_REQUEST_BYTES:
        return None, f'finish request at {path} is larger than {MAX_REQUEST_BYTES} bytes; continuing to run'
    return blob, ''


def _parse(path: Path, blob: bytes) -> tuple[dict[str, Any] | None, str]:
    """The request object those bytes hold, or `None` and the reason they hold none."""
    try:
        request = json.loads(blob.decode())
    except MALFORMED_CONTENT as e:
        return None, f'finish request at {path} could not be read as a request ({e!r}); continuing to run'
    if not isinstance(request, dict):
        return None, f'finish request at {path} is {type(request).__name__}, not an object; continuing to run'
    return request, ''


def _asks_this_run_to_finish(path: Path, request: dict[str, Any], this_run: str) -> tuple[bool, str]:
    """Whether the request object asks `this_run` to finish, and why not when it does not.

    Runs outside every handler: the input is a `dict` by here, so anything raising is a defect and
    ends the World rather than reading as a file that was never a request.
    """
    named = request.get(ACTION_KEY)
    # The type is checked before the value because JSON's containers are unhashable: an array or
    # object reaching the membership test below raises `TypeError`, and nothing here catches one —
    # a file somebody wrote would end the run as though this code were broken. An absent key arrives
    # as `None` and is the same fact: nothing an action reads from.
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
        # The grant, latched: once made it is never withdrawn, and it outlives the World that read it.
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
