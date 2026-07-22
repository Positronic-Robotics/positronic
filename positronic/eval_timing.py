"""Opt-in wall-clock telemetry for ``positronic eval run``.

A sim eval runs on a virtual clock, so the *virtual* time it advances (recorded in the episode's
signal timestamps) says nothing about how much real compute a rollout cost. This module captures the
wall-clock split a sizing/perf pass needs — policy inference, env reset, env step, record IO — that the
recorded dataset cannot recover on its own. Everything the dataset *can* recover (episode duration,
success, on-disk size) is left to the reduce step (`positronic.cli.eval.timing_report`) to join back in,
so nothing is stored twice.

The collector is bound around an eval sweep via a ``ContextVar`` (`bind`) and reached at the hook sites
through this module's hook functions (`begin_episode`, `record_infer`, `timed`, ...), each a no-op while
unbound — the default — so a normal eval pays nothing.
Because a sim eval schedules the harness, recorder and env proxy as cooperative generators in one thread,
a single bound collector sees all of their timings. A real eval runs the recorder and producers as
separate processes, which do not inherit the context — so this telemetry is sim-only by construction.
"""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import IntEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

TIMING_FILENAME = 'timing.jsonl'
GPU_LOG_FILENAME = 'gpu_dmon.log'


class Phase(IntEnum):
    """A rollout span timed via ``timed``; it accumulates into the matching ``EpisodeTiming`` field."""

    RESET = auto()
    ENV_STEP = auto()
    MATERIALIZE = auto()  # client-side observation assembly — part of the env step, tracked apart from wire
    RECORD_IO = auto()


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

    ``env_physics_s``/``env_render_s``/``env_server_s`` are the env server's own decomposition of the time
    inside ``env_step_s`` — physics substeps, sensor rendering, and the server's whole in-step wall.
    ``env_client_s`` is the client-side observation materialisation (shared-memory image allocation + camera
    copies) that also runs inside ``env_step_s``, so ``env_step_s - env_server_s - env_client_s`` is the wire
    + codec cost. All the env-decomposition fields are zero for envs that report none.

    ``finished_at`` is the wall clock (epoch seconds) at which the episode was sealed; with ``wall_s`` it lets
    the reduce recover the pass span — the first episode's start to the last's finish — so inter-episode
    teardown wall counts in ``W_pass`` instead of being dropped by a per-episode sum.
    """

    episode_uid: str
    wall_s: float
    reset_s: float
    env_step_s: float
    policy_wait_s: float
    record_io_s: float
    overhead_s: float
    infer_ms: list[float]
    finished_at: float
    env_physics_s: float = 0.0
    env_render_s: float = 0.0
    env_server_s: float = 0.0
    env_client_s: float = 0.0


@dataclass
class _Accumulator:
    """The running sums for the episode in flight; sealed into an ``EpisodeTiming`` at finish."""

    wall_start: float
    reset_s: float = 0.0
    env_step_s: float = 0.0
    policy_wait_s: float = 0.0
    record_io_s: float = 0.0
    env_physics_s: float = 0.0
    env_render_s: float = 0.0
    env_server_s: float = 0.0
    env_client_s: float = 0.0
    infer_ms: list[float] = field(default_factory=list)


class EvalTimer:
    """Collects one ``EpisodeTiming`` per rollout, appending each to ``timing.jsonl`` as it seals.

    The harness opens each episode (`begin_episode`) and the recorder closes it (`finish_episode`) — the
    same single thread, in order — so at most one episode is ever in flight. Records are flushed per episode,
    not buffered, so a preempted or killed pass keeps the timings of the episodes it completed.
    """

    def __init__(self, out_dir: Path):
        self._path = out_dir / TIMING_FILENAME
        self._path.write_text('')  # truncate any stale pass; episodes append as they finish
        self._count = 0
        self._current: _Accumulator | None = None

    def begin_episode(self) -> None:
        if self._current is not None:
            logger.warning('EvalTimer: new episode began before the previous one finished; dropping the previous')
        self._current = _Accumulator(wall_start=time.perf_counter())

    def _record(self, phase: Phase, seconds: float) -> None:
        """Accumulate a timed span into the in-flight episode. ``MATERIALIZE`` is part of the env step and
        also tracked apart (``env_client_s``) so the reduce can split it out of the wire cost."""
        if self._current is None:
            return
        match phase:
            case Phase.RESET:
                self._current.reset_s += seconds
            case Phase.ENV_STEP:
                self._current.env_step_s += seconds
            case Phase.MATERIALIZE:
                self._current.env_step_s += seconds
                self._current.env_client_s += seconds
            case Phase.RECORD_IO:
                self._current.record_io_s += seconds

    def add_infer(self, seconds: float) -> None:
        """One policy round-trip: it is the whole policy wait, and one entry in the latency distribution."""
        if self._current is not None:
            self._current.policy_wait_s += seconds
            self._current.infer_ms.append(seconds * 1000.0)

    def add_env_phases(self, physics_s: float, render_s: float, server_s: float) -> None:
        """One env step's server-reported decomposition: physics substeps, rendering, whole in-step wall."""
        if self._current is not None:
            self._current.env_physics_s += physics_s
            self._current.env_render_s += render_s
            self._current.env_server_s += server_s

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
        record = EpisodeTiming(
            episode_uid=episode_uid,
            wall_s=wall_s,
            reset_s=acc.reset_s,
            env_step_s=acc.env_step_s,
            policy_wait_s=acc.policy_wait_s,
            record_io_s=acc.record_io_s,
            overhead_s=max(wall_s - measured, 0.0),
            infer_ms=[round(ms, 3) for ms in acc.infer_ms],
            finished_at=time.time(),
            env_physics_s=acc.env_physics_s,
            env_render_s=acc.env_render_s,
            env_server_s=acc.env_server_s,
            env_client_s=acc.env_client_s,
        )
        # Append and flush as the episode seals, so a pass killed mid-run keeps every completed rollout's
        # timing instead of losing an in-memory buffer.
        with self._path.open('a') as f:
            f.write(json.dumps(asdict(record)) + '\n')
        self._count += 1

    def close(self) -> Path:
        """Return the ``timing.jsonl`` path; records were already flushed as each episode sealed."""
        logger.info(f'EvalTimer: wrote {self._count} episode timings to {self._path}')
        return self._path


_ACTIVE: ContextVar[EvalTimer | None] = ContextVar('eval_timer', default=None)

# Hook sites call these module functions, never the timer directly — each is a no-op when telemetry is off,
# so a normal eval pays nothing and no site carries a ``None`` check.


def begin_episode() -> None:
    """Open a new rollout's timing span."""
    if (timer := _ACTIVE.get()) is not None:
        timer.begin_episode()


def finish_episode(episode_uid: str) -> None:
    """Seal the in-flight rollout under its recorded uid."""
    if (timer := _ACTIVE.get()) is not None:
        timer.finish_episode(episode_uid)


def discard_episode() -> None:
    """Drop the in-flight rollout without recording it — an abort."""
    if (timer := _ACTIVE.get()) is not None:
        timer.discard_episode()


def record_infer(seconds: float) -> None:
    """Record one policy round-trip's wall time (whole policy wait + one latency sample)."""
    if (timer := _ACTIVE.get()) is not None:
        timer.add_infer(seconds)


def record_env_phases(physics_s: float, render_s: float, server_s: float) -> None:
    """Record an env step's server-reported physics/render/whole-wall decomposition."""
    if (timer := _ACTIVE.get()) is not None:
        timer.add_env_phases(physics_s, render_s, server_s)


def _start_gpu_sampler(out_dir: Path) -> subprocess.Popen | None:
    """Background ``nvidia-smi dmon`` writing this box's util+memory to ``gpu_dmon.log``.

    ``None`` when no ``nvidia-smi`` is on PATH (a CPU dev box) — GPU telemetry is then simply absent, not
    an error. ``-i`` pins the one GPU this eval uses (``-s um`` samples SM/memory utilisation + framebuffer,
    ``-d 1`` once a second, ``-o DT`` prefixes each row with date and time).
    """
    # A prior pass in the same output_dir may have left a log; drop it first so a run that ends up without
    # GPU samples (no nvidia-smi here) can't be summarised against a stale one — the reducer auto-reads
    # this file whenever it exists.
    log_path = out_dir / GPU_LOG_FILENAME
    log_path.unlink(missing_ok=True)
    if shutil.which('nvidia-smi') is None:
        logger.info('EvalTimer: no nvidia-smi on PATH; skipping GPU sampling')
        return None
    # Sample only the GPU this eval runs on — the first CUDA-visible device, else device 0. Left unpinned,
    # dmon logs every visible GPU and ``_parse_dmon`` would average idle/unrelated devices into the numbers.
    device = (os.environ.get('CUDA_VISIBLE_DEVICES', '') or '0').split(',')[0]
    return subprocess.Popen([
        'nvidia-smi',
        'dmon',
        '-i',
        device,
        '-s',
        'um',
        '-d',
        '1',
        '-o',
        'DT',
        '-f',
        str(log_path),
    ])


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
def timed(phase: Phase) -> Iterator[None]:
    """Time the enclosed block into ``phase``. A no-op when telemetry is off, so a normal eval pays nothing."""
    timer = _ACTIVE.get()
    if timer is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timer._record(phase, time.perf_counter() - start)
