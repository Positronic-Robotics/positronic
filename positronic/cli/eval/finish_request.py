"""Ending an attended run from outside it, without a signal.

An attended run is ended by its operator surface: the web console POSTs `/finish_run` to itself.
A run driven by the local eval UI has no such surface — it serves no port — so the only way to end
one from outside the process was a signal, and a signal is not a finish: nothing unwinds the World,
so the recording is not closed, the dataset is not uploaded, and the arm is left as it stands with
its control token held.

This is the surface an orchestrator uses instead. It writes a small file; the run polls it and, once
the episode in progress has completed, stops the way a plan running out stops. What follows is the
ordinary shutdown — the World unwinds, the recorder commits, the mirror uploads on process exit, and
the arm's driver releases control.

THE CONTRACT, which a writer in another repository implements against:

  path     `<dir>/positronic_rollout_finish.<run_id>`, where `<dir>` is `/run/lock` unless
           `ROLLOUT_FINISH_REQUEST_DIR` names another. Absolute, because the writer and the run are
           different accounts on a rig that gives each actor its own home — a `$HOME`-relative path
           is a different file per actor and reaches nobody. On tmpfs, so a request cannot outlive a
           reboot, and in a world-writable sticky directory, so neither account needs root or a
           shared group.

           PER RUN, which is what makes a leftover inert. One rig-wide path is a shared mutable
           cell: a file left by a run that has ended names a run nobody will start again, yet in a
           sticky directory only its creating account may replace it, so the next run under another
           account cannot write its own request and has no clean exit at all. Scoped by run id,
           every request is addressed to exactly one reader, no reader ever sees another's, and
           removal stops being part of correctness — it is hygiene, and the reboot that clears the
           tmpfs is the only cleanup anyone needs.

           `run_id` is therefore ONE PATH SEGMENT: no separator, and not `.` or `..`. A run named
           otherwise installs no poller rather than reading a path it did not mean.
  content  one JSON object: `{"action": "finish", "run_id": "<id>"}`. Further keys are ignored, so a
           writer may record its own diagnostics (when it asked, who asked) without a format change.
  writer   creates it, world-readable (mode 0644 — a `077` umask would leave it 0600 and unreadable
           by the run, which is silent). The run never writes or unlinks it: the sticky bit means
           only the owner can, so a run that tried would fail every time.
  reader   this module, in the run's process, read-only.
  identity `run_id` must equal the run's own `ROLLOUT_RUN_ID`. The path already names it, and the
           content is checked anyway: the two disagree only where a file was copied or a directory
           override collided, and a run stopped by a request that was never addressed to it is
           indistinguishable afterwards from one that finished early on its own.
  intent   MONOTONIC. A request, once made, is never withdrawn — there is no un-ask, and no content
           that means "carry on". A writer whose write may or may not have landed therefore
           re-asserts it rather than retracting it, and a reader needs no protocol for a request
           that changes its mind. It is why the file can be a rendering of the writer's own record
           rather than the record itself: asserting it twice is asserting it once.

It FAILS CLOSED, where closed means the run keeps running: an absent file, an unreadable one, one
that does not parse, one naming another run, and one whose action is not `finish` are all ignored,
each logged once. The failure this ordering prevents is a run stopped by something that was never a
request — the mirror of the writer's own problem, which is a request that is never picked up, and
which the writer answers with a bounded wait rather than by assuming this side acted.

Nothing is installed when `ROLLOUT_RUN_ID` is unset, so an ordinary local run is untouched.
"""

import json
import logging
import os
import time
from pathlib import Path

import pimm

logger = logging.getLogger(__name__)

# The directory requests are written into, and the name each one takes inside it. Overridable so a
# test, or a rig running more than one simulated run, can use a directory of its own rather than the
# one tmpfs every run on a machine shares.
FINISH_REQUEST_DIR_ENV = 'ROLLOUT_FINISH_REQUEST_DIR'
DEFAULT_FINISH_REQUEST_DIR = '/run/lock'
FINISH_REQUEST_PREFIX = 'positronic_rollout_finish.'
# The run's own identity, set by whatever launched it. Its absence is what makes this inert.
RUN_ID_ENV = 'ROLLOUT_RUN_ID'

# The object's two required fields and the only action understood. A second action (discard the open
# episode and stop) would be a new value here rather than a second file.
ACTION_KEY = 'action'
RUN_ID_KEY = 'run_id'
FINISH_ACTION = 'finish'

# How often the file is read. The wait it adds to a finish is bounded by this, and the cost of it is
# one `open` on tmpfs, so it is short enough not to be noticed and long enough not to matter.
POLL_INTERVAL_S = 2.0

# Printed to the run's log the moment a request is granted, and part of the contract: it is the only
# way the writer can tell a request still being worked through — the run is mid-episode, which can
# take minutes — from one that never arrived at all, because the path is wrong, the file is
# unreadable, or this run predates the poller. Those need opposite responses from whoever asked, and
# from outside the process both look like a run that is still running.
ACK_LOG_MARKER = 'rollout finish request granted'


def request_dir() -> Path:
    return Path(os.environ.get(FINISH_REQUEST_DIR_ENV, DEFAULT_FINISH_REQUEST_DIR))


def request_path(this_run: str) -> Path:
    """Where a request addressed to `this_run` is read from — the whole of the path contract."""
    return request_dir() / f'{FINISH_REQUEST_PREFIX}{this_run}'


def names_one_segment(this_run: str) -> bool:
    """Whether `this_run` can be a filename, which is what scoping the path by run id requires.

    A run id carrying a separator would address a file in a directory nobody agreed on — under the
    request directory for a relative one, and anywhere at all for `..` — so a run named that way is
    left with no poller rather than polling somewhere unintended.
    """
    return bool(this_run) and this_run not in ('.', '..') and '/' not in this_run and '\0' not in this_run


def run_id() -> str | None:
    """This run's identity, or None where nothing gave it one."""
    return os.environ.get(RUN_ID_ENV) or None


def evaluate(path: Path, this_run: str) -> tuple[bool, str]:
    """Whether `path` holds a finish request addressed to `this_run`, and why not when it does not.

    Every negative is a reason to keep running, so the caller acts on the bool alone; the reason is
    for the log, because "the request is there and the run ignored it" and "the request never
    arrived" look identical from the writer's side and have different fixes. It is returned rather
    than logged here so the caller can report a persistent one once instead of on every poll —
    a request addressed to a dead run would otherwise fill the log for the life of this one.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return False, ''
    except OSError as e:
        # Unreadable is not absent: a request written by another account with a restrictive umask
        # lands here, and it is the one negative a reader cannot fix by waiting.
        return False, f'finish request at {path} could not be read ({e}); continuing to run'
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f'finish request at {path} did not parse ({e}); continuing to run'
    if not isinstance(request, dict):
        return False, f'finish request at {path} is {type(request).__name__}, not an object; continuing to run'
    if request.get(ACTION_KEY) != FINISH_ACTION:
        action = request.get(ACTION_KEY)
        return False, f'finish request at {path} names action {action!r}, not {FINISH_ACTION!r}; continuing to run'
    addressee = request.get(RUN_ID_KEY)
    if addressee != this_run:
        # The stale-request case, and the only one that is routine rather than a fault: a previous
        # run's request outliving it.
        return False, f'finish request at {path} names run {addressee!r}, not {this_run!r}; continuing to run'
    return True, ''


class FinishRequest(pimm.ControlSystem):
    """Emits True once a finish request addressed to this run appears, and keeps emitting it.

    It never finishes its own loop. A control system that returns sets the world's stop event, which
    would end the run wherever it happened to be — the mid-episode stop this exists to replace. The
    harness owns when it is safe to stop; this only reports that someone asked.
    """

    def __init__(self, path: Path, this_run: str, poll_interval_s: float = POLL_INTERVAL_S):
        self.requested = pimm.ControlSystemEmitter[bool](self)
        self._path = path
        self._run = this_run
        self._poll_interval_s = poll_interval_s
        # The last refusal reported, so a request that stays wrong is logged when it appears and not
        # on every poll for the life of the run. Reported again when the reason CHANGES, which is what
        # a replaced file looks like from here.
        self._reported = ''

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        # Paced on the WALL clock, not the world's. Under virtual time the world runs as fast as the
        # machine allows, so a `Sleep` of two simulated seconds is no wait at all and this would read
        # the file on a tight loop.
        last_read = 0.0
        granted = False
        while not should_stop.value:
            if not granted and time.monotonic() - last_read >= self._poll_interval_s:
                last_read = time.monotonic()
                granted, reason = evaluate(self._path, self._run)
                if granted:
                    logger.info('%s: run %s stops after the current episode', ACK_LOG_MARKER, self._run)
                elif reason != self._reported:
                    self._reported = reason
                    if reason:
                        logger.error(reason)
            if granted:
                # Re-emitted every round rather than once: the harness reads a signal, and a single
                # emit that landed before it bound would be a request that silently never arrives.
                self.requested.emit(True, clock.now_ns())
            yield pimm.Sleep(self._poll_interval_s)


def from_env() -> FinishRequest | None:
    """The control system this run should carry, or None where nothing named the run."""
    this_run = run_id()
    if this_run is None:
        return None
    if not names_one_segment(this_run):
        logger.error(
            '%s=%r is not a single path segment, so no finish request can address this run', RUN_ID_ENV, this_run
        )
        return None
    path = request_path(this_run)
    logger.info('run %s will stop after the current episode if %s asks it to', this_run, path)
    return FinishRequest(path, this_run)
