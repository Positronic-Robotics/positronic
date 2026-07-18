"""Opt-in wall-clock telemetry for ``positronic eval run``.

A sim eval runs on a virtual clock, so the *virtual* time it advances (recorded in the episode's
signal timestamps) says nothing about how much real compute a rollout cost. This module captures the
wall-clock split a sizing/perf pass needs — policy inference, env reset, env step, record IO — that the
recorded dataset cannot recover on its own. Everything the dataset *can* recover (episode duration,
success, on-disk size) is left to the reduce step (`positronic.cli.eval.timing_report`) to join back in,
so nothing is stored twice.

The collector is bound for the span of one ``World`` run via a ``ContextVar`` (`bind`) and read at the
hook sites with `active`; unbound — the default — every hook is a no-op, so a normal eval pays nothing.
Because a sim eval schedules the harness, recorder and env proxy as cooperative generators in one thread,
a single bound collector sees all of their timings. A real eval runs the recorder and producers as
separate processes, which do not inherit the context — so this telemetry is sim-only by construction.
"""

import contextlib
import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TIMING_FILENAME = 'timing.jsonl'
GPU_LOG_FILENAME = 'gpu_dmon.log'


@dataclass
class EpisodeTiming:
    """The wall-clock costs of one rollout, in seconds unless the name says otherwise.

    ``overhead_s`` is the wall time unattributed to the measured phases (pimm scheduling, observation
    assembly, command demux). Under a virtual clock the phases sum close to ``wall_s`` because pimm's
    pacing sleeps advance virtual time without consuming wall time, so a large overhead is itself a
    signal worth reading. ``infer_ms`` is the raw per-call policy round-trip latency — kept raw, not
    pre-summarised, so the reduce step derives exact pass-level percentiles rather than pooling
    per-episode ones (a whole pass is only ~10-20k calls, so the raw list stays small). ``episode_uid``
    is the key the reduce step joins on to pull duration, success and byte size from the recorded episode.
    """

    task: str
    trial: int
    episode_uid: str
    wall_s: float
    reset_s: float
    env_step_s: float
    policy_wait_s: float
    record_io_s: float
    overhead_s: float
    infer_ms: list[float]


@dataclass
class _Accumulator:
    """The running sums for the episode in flight; sealed into an ``EpisodeTiming`` at finish."""

    task: str
    trial: int
    wall_start: float
    reset_s: float = 0.0
    env_step_s: float = 0.0
    policy_wait_s: float = 0.0
    record_io_s: float = 0.0
    infer_ms: list[float] = field(default_factory=list)


class EvalTimer:
    """Collects one ``EpisodeTiming`` per rollout and writes them as ``timing.jsonl`` on close.

    The harness opens each episode (`begin_episode`) and the recorder closes it (`finish_episode`) — the
    same single thread, in order — so at most one episode is ever in flight.
    """

    def __init__(self, out_dir: Path):
        self._out_dir = out_dir
        self._records: list[EpisodeTiming] = []
        self._current: _Accumulator | None = None

    def begin_episode(self, task: str, trial: int) -> None:
        if self._current is not None:
            logger.warning('EvalTimer: new episode began before the previous one finished; dropping the previous')
        self._current = _Accumulator(task=task, trial=trial, wall_start=time.perf_counter())

    def add_reset(self, seconds: float) -> None:
        if self._current is not None:
            self._current.reset_s += seconds

    def add_env_step(self, seconds: float) -> None:
        if self._current is not None:
            self._current.env_step_s += seconds

    def add_infer(self, seconds: float) -> None:
        """One policy round-trip: it is the whole policy wait, and one entry in the latency distribution."""
        if self._current is not None:
            self._current.policy_wait_s += seconds
            self._current.infer_ms.append(seconds * 1000.0)

    def add_record_io(self, seconds: float) -> None:
        if self._current is not None:
            self._current.record_io_s += seconds

    def discard_episode(self) -> None:
        """Drop the in-flight episode without recording it (an abort)."""
        self._current = None

    def finish_episode(self, episode_uid: str) -> None:
        acc = self._current
        if acc is None:
            return
        self._current = None
        wall_s = time.perf_counter() - acc.wall_start
        measured = acc.reset_s + acc.env_step_s + acc.policy_wait_s + acc.record_io_s
        self._records.append(
            EpisodeTiming(
                task=acc.task,
                trial=acc.trial,
                episode_uid=episode_uid,
                wall_s=wall_s,
                reset_s=acc.reset_s,
                env_step_s=acc.env_step_s,
                policy_wait_s=acc.policy_wait_s,
                record_io_s=acc.record_io_s,
                overhead_s=max(wall_s - measured, 0.0),
                infer_ms=[round(ms, 3) for ms in acc.infer_ms],
            )
        )

    def close(self) -> Path:
        """Write every collected record as one JSON object per line; return the file path."""
        path = self._out_dir / TIMING_FILENAME
        with path.open('w') as f:
            for record in self._records:
                f.write(json.dumps(asdict(record)) + '\n')
        logger.info(f'EvalTimer: wrote {len(self._records)} episode timings to {path}')
        return path


_ACTIVE: ContextVar[EvalTimer | None] = ContextVar('eval_timer', default=None)


def active() -> EvalTimer | None:
    """The timer bound for the current ``World`` run, or ``None`` when telemetry is off."""
    return _ACTIVE.get()


def _start_gpu_sampler(out_dir: Path) -> subprocess.Popen | None:
    """Background ``nvidia-smi dmon`` writing this box's util+memory to ``gpu_dmon.log``.

    ``None`` when no ``nvidia-smi`` is on PATH (a CPU dev box) — GPU telemetry is then simply absent, not
    an error. ``-s um`` samples SM/memory utilisation + framebuffer, ``-d 1`` once a second, ``-o DT``
    prefixes each row with date and time.
    """
    if shutil.which('nvidia-smi') is None:
        logger.info('EvalTimer: no nvidia-smi on PATH; skipping GPU sampling')
        return None
    log_path = out_dir / GPU_LOG_FILENAME
    return subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'um', '-d', '1', '-o', 'DT', '-f', str(log_path)])


@contextlib.contextmanager
def bind(out_dir: Path) -> Iterator[EvalTimer]:
    """Bind a fresh timer (and a GPU sampler) for the enclosed run, then flush ``timing.jsonl`` on exit."""
    timer = EvalTimer(out_dir)
    token = _ACTIVE.set(timer)
    sampler = _start_gpu_sampler(out_dir)
    try:
        yield timer
    finally:
        _ACTIVE.reset(token)
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
        timer.close()


@contextlib.contextmanager
def timed(sink: Callable[[float], None] | None) -> Iterator[None]:
    """Feed the wall duration of the enclosed block to ``sink``; a no-op when ``sink`` is ``None``."""
    if sink is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        sink(time.perf_counter() - start)
