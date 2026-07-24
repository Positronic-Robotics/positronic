"""Opt-in wall-clock telemetry for ``positronic eval run``.

A sim eval runs on a virtual clock, so the *virtual* time it advances (recorded in the episode's
signal timestamps) says nothing about how much real compute a rollout cost. This module captures the
wall-clock split a sizing/perf pass needs — policy inference, env reset, env step, record IO — that the
virtual-clock timestamps cannot recover, and hands it to the recorder to store *in the episode itself*:
the per-tick phase costs and per-call inference latencies as ``timing.*`` signals, and the once-per-episode
wall/finish/reset scalars as statics. The pass-level roll-up is then an offline reduce over those recorded
raw values (`positronic.cli.eval.timing_report`) — nothing is stored twice and no side channel is written.

The collector is bound around an eval sweep via a ``ContextVar`` (`bind`) and reached at the hook sites
through this module's hook functions (`begin_episode`, `record_infer`, `timed`, ...), each a no-op while
unbound — the default — so a normal eval pays nothing.
Because a sim eval schedules the harness, recorder and env proxy as cooperative generators in one thread,
a single bound collector sees all of their timings. A real eval runs the recorder and producers as
separate processes, which do not inherit the context — so this telemetry is sim-only by construction.
"""

import contextlib
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

GPU_LOG_FILENAME = 'gpu_dmon.log'

# The name prefix under which this module's telemetry signals and statics live — a producer-side naming
# convention this module owns, not something the dataset core interprets or dispatches on.
TELEMETRY_PREFIX = 'timing.'

# The episode signal + static keys the recorder writes and the reduce step reads back — one source of truth
# so the writer and `timing_report` never drift on a name. All share the ``TELEMETRY_PREFIX`` namespace: the
# per-tick phase costs and the per-call inference latencies as signals, the once-per-episode wall scalars as
# statics.
INFER_MS_SIGNAL = f'{TELEMETRY_PREFIX}infer_ms'
WALL_S_KEY = f'{TELEMETRY_PREFIX}wall_s'
FINISHED_AT_KEY = f'{TELEMETRY_PREFIX}finished_at'
RESET_S_KEY = f'{TELEMETRY_PREFIX}reset_s'


class Phase(IntEnum):
    """A rollout span timed via ``timed``; it accumulates into the matching timing field."""

    RESET = auto()
    ENV_STEP = auto()
    MATERIALIZE = auto()  # client-side observation assembly — part of the env step, tracked apart from wire
    RECORD_IO = auto()


@dataclass
class StepTiming:
    """The per-tick wall-clock costs accumulated between two recorder drains, in seconds. Its field names are
    the ``timing.*`` signal suffixes the recorder appends, so the reduce step sums each column back to an
    episode total.

    ``env_step_s`` includes the client-side materialisation; ``env_client_s`` is that materialisation alone
    (shared-memory image allocation + camera copies), so ``env_step_s - env_server_s - env_client_s`` is the
    wire + codec cost. ``env_physics_s``/``env_render_s``/``env_server_s`` are the env server's own
    decomposition — physics substeps, sensor rendering, and its whole in-step wall — and are zero for envs
    that report none.
    """

    env_step_s: float = 0.0
    record_io_s: float = 0.0
    env_physics_s: float = 0.0
    env_render_s: float = 0.0
    env_server_s: float = 0.0
    env_client_s: float = 0.0

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) == 0.0 for f in fields(self))


@dataclass
class EpisodeTimingStatics:
    """The per-episode wall scalars the recorder stores as statics alongside the ``timing.*`` signals.

    ``reset_s`` is the once-per-episode env reset span, which is not part of the per-tick rollout loop.
    ``wall_s`` is the whole rollout wall and ``finished_at`` the epoch at which it sealed; together they let
    the reduce recover each episode's start and so the pass span (first start to last finish), counting
    inter-episode teardown that a per-episode sum would drop.
    """

    wall_s: float
    finished_at: float
    reset_s: float


@dataclass
class _Episode:
    """The in-flight rollout: its wall start, the reset span, the un-drained step, and the un-drained
    per-call inference latencies (ms)."""

    wall_start: float  # perf_counter at begin
    step: StepTiming = field(default_factory=StepTiming)
    reset_s: float = 0.0
    pending_infer_ms: list[float] = field(default_factory=list)


class EvalTimer:
    """Collects one rollout's wall-clock costs and hands them to the recorder to store in the episode.

    The harness opens each episode (`begin_episode`); the recorder drains the accumulated per-tick costs and
    inference latencies once per control tick (`drain_step`/`drain_infers`) and seals the episode at STOP
    (`finish_episode`) — the same single thread, in order — so at most one episode is ever in flight.
    """

    def __init__(self) -> None:
        self._current: _Episode | None = None

    def begin_episode(self) -> None:
        if self._current is not None:
            logger.warning('EvalTimer: new episode began before the previous one finished; dropping the previous')
        self._current = _Episode(wall_start=time.perf_counter())

    def _record(self, phase: Phase, seconds: float) -> None:
        """Accumulate a timed span. ``RESET`` is the once-per-episode reset; the rest fold into the in-flight
        step. ``MATERIALIZE`` is part of the env step and also tracked apart (``env_client_s``) so the reduce
        can split it out of the wire cost."""
        if self._current is None:
            return
        step = self._current.step
        match phase:
            case Phase.RESET:
                self._current.reset_s += seconds
            case Phase.ENV_STEP:
                step.env_step_s += seconds
            case Phase.MATERIALIZE:
                step.env_step_s += seconds
                step.env_client_s += seconds
            case Phase.RECORD_IO:
                step.record_io_s += seconds

    def add_infer(self, seconds: float) -> None:
        """One policy round-trip: one pending sample of the per-call inference-latency signal (whole wait)."""
        if self._current is not None:
            self._current.pending_infer_ms.append(seconds * 1000.0)

    def add_env_phases(self, physics_s: float, render_s: float, server_s: float) -> None:
        """One env step's server-reported decomposition: physics substeps, rendering, whole in-step wall."""
        if self._current is not None:
            step = self._current.step
            step.env_physics_s += physics_s
            step.env_render_s += render_s
            step.env_server_s += server_s

    def drain_step(self) -> StepTiming | None:
        """Return the per-tick costs accumulated since the last drain and reset the step. ``None`` when no
        episode is in flight — the recorder then appends no timing sample this tick."""
        if self._current is None:
            return None
        step = self._current.step
        self._current.step = StepTiming()
        return step

    def drain_infers(self) -> list[float]:
        """Return the policy round-trip latencies (ms) recorded since the last drain and clear them — each is
        one sample of the per-call ``timing.infer_ms`` signal. Empty when no episode is in flight."""
        if self._current is None:
            return []
        infers = self._current.pending_infer_ms
        self._current.pending_infer_ms = []
        return infers

    def discard_episode(self) -> None:
        """Drop the in-flight episode without sealing it (an abort)."""
        self._current = None

    def finish_episode(self) -> EpisodeTimingStatics | None:
        """Seal the in-flight rollout and return its per-episode statics. ``None`` when none is in flight."""
        acc = self._current
        if acc is None:
            return None
        self._current = None
        return EpisodeTimingStatics(
            wall_s=time.perf_counter() - acc.wall_start, finished_at=time.time(), reset_s=acc.reset_s
        )


_ACTIVE: ContextVar[EvalTimer | None] = ContextVar('eval_timer', default=None)

# Hook sites call these module functions, never the timer directly — each is a no-op when telemetry is off,
# so a normal eval pays nothing and no site carries a ``None`` check.


def begin_episode() -> None:
    """Open a new rollout's timing span."""
    if (timer := _ACTIVE.get()) is not None:
        timer.begin_episode()


def discard_episode() -> None:
    """Drop the in-flight rollout without sealing it — an abort."""
    if (timer := _ACTIVE.get()) is not None:
        timer.discard_episode()


def record_infer(seconds: float) -> None:
    """Record one policy round-trip's wall time (one latency sample)."""
    if (timer := _ACTIVE.get()) is not None:
        timer.add_infer(seconds)


def record_env_phases(physics_s: float, render_s: float, server_s: float) -> None:
    """Record an env step's server-reported physics/render/whole-wall decomposition."""
    if (timer := _ACTIVE.get()) is not None:
        timer.add_env_phases(physics_s, render_s, server_s)


def drain_signal_items() -> Iterator[tuple[str, float]]:
    """One tick's drained wall-clock telemetry as ``(signal_name, seconds)`` pairs: each non-zero phase cost
    as its ``timing.<phase>`` signal, then each policy round-trip as one ``timing.infer_ms`` sample. Empty
    when telemetry is off or the tick accumulated nothing — the recorder appends whatever it yields, without
    knowing what the names mean."""
    timer = _ACTIVE.get()
    if timer is None:
        return
    step = timer.drain_step()
    if step is not None and not step.is_empty():
        for name, seconds in asdict(step).items():
            if seconds != 0.0:
                yield f'{TELEMETRY_PREFIX}{name}', seconds
    for infer_ms in timer.drain_infers():
        yield INFER_MS_SIGNAL, infer_ms


def finish_static_items() -> Iterator[tuple[str, object]]:
    """Seal the in-flight rollout and yield the ``(static_key, value)`` pairs the recorder sets on it — the
    per-episode wall scalars. Empty when telemetry is off or no rollout is in flight."""
    timer = _ACTIVE.get()
    if timer is None:
        return
    statics = timer.finish_episode()
    if statics is not None:
        yield WALL_S_KEY, statics.wall_s
        yield FINISHED_AT_KEY, statics.finished_at
        yield RESET_S_KEY, statics.reset_s


class WriterHooks:
    """The recorder's opaque view of this module's telemetry: per-tick ``(name, value)`` drains, a record-IO
    span, and the seal/discard lifecycle. Every method is inert when no collector is bound (a normal eval), so
    the recorder carries one unconditionally and pays nothing off the timing path."""

    def record_io(self) -> contextlib.AbstractContextManager:
        return timed(Phase.RECORD_IO)

    def drain(self) -> Iterator[tuple[str, float]]:
        return drain_signal_items()

    def finish(self) -> Iterator[tuple[str, object]]:
        return finish_static_items()

    def discard(self) -> None:
        discard_episode()


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
    """Bind a fresh collector (and a GPU sampler) for the enclosed run.

    The GPU sampler's ``gpu_dmon.log`` is the one telemetry stream the dataset cannot carry — a background
    per-box time series that outlives any single episode — so it stays a side file the reduce reads; the
    per-rollout timing rides the recorded episodes.
    """
    timer = EvalTimer()
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
