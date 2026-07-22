"""Emit ``positronic eval timing-report``-compatible telemetry from a rollout loop positronic does not run.

Some sims (MolmoSpaces is the first) run their own rollout loop and know nothing about positronic's dataset
format, yet a sizing pass wants the same pass-level wall-clock report ``positronic eval timing-report``
produces for native evals. This shim is the adapter: the harness wraps it around the rollout loop, marks each
phase (policy wait, env step, reset, record IO) and each episode's outcome, and the shim writes both halves
the report needs — the ``timing.jsonl`` wall-clock records AND the recorded-dataset stubs (block/episode dirs
with ``meta.json`` + ``static.json`` + a timestamped signal) the report joins them against by ``episode_uid``.
Optional ``nvidia-smi dmon`` sampling folds in GPU utilisation and peak VRAM, and is simply absent on a CPU
box. A lightweight host sampler additionally logs per-second CPU (``/proc/stat``) and RAM (``/proc/meminfo``)
to ``host_stats.log`` on any box, GPU or CPU; the reducer ignores that sidecar (it reads only
``timing.jsonl``, ``gpu_dmon.log`` and the recorded episode dirs), so it is a pass-time telemetry artifact,
not a join input.

The emitted layout is exactly what the reducer consumes::

    out_dir/
      timing.jsonl                 # one EpisodeTiming JSON object per rollout
      gpu_dmon.log                 # optional nvidia-smi dmon -s um log (sim box)
      host_stats.log               # optional per-interval host CPU/RAM samples (any box)
      000000000000/                # 12-digit block dir = (episode_id // 1000) * 1000
        000000000000/              # 12-digit episode dir = episode_id
          meta.json                # uid (the join key), created_ts_ns, duration_ns, size_mb
          static.json              # eval.success / eval.terminated / eval.scored
          rollout.parquet          # one timestamped signal, so duration derives on its own too

Usage::

    shim = TimingShim(out_dir, task='pick_cube')
    with shim.run():
        for trial in range(n_trials):
            with shim.episode(trial=trial, sim_duration_s=n_steps * control_dt) as ep:
                with ep.reset():
                    obs = env.reset()
                for _ in range(n_steps):
                    with ep.policy():
                        action = policy(obs)
                    with ep.env_step():
                        obs, info = env.step(action)
                    ep.add_env_phases(info['physics_s'], info['render_s'], info['server_s'])
                    with ep.record_io():
                        recorder.write(obs)
                ep.set_outcome(success=bool(info['success']), terminated=True, scored=True)

Every ``ep.<phase>()`` block is a context manager that times its body; the ``add_*`` methods are the
equivalent callbacks for phases whose wall time the harness already measured. ``run()`` and ``episode()``
own the pass and per-episode lifecycles; both are optional wrappers over the explicit ``start_gpu`` /
``close`` and ``begin_episode`` / ``finish_episode`` calls beneath them.
"""

import contextlib
import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from positronic.eval_timing import TIMING_FILENAME, EpisodeTiming, start_gpu_sampler

logger = logging.getLogger(__name__)

HOST_LOG_FILENAME = 'host_stats.log'
SIGNAL_FILENAME = 'rollout.parquet'


@dataclass
class Episode:
    """One in-flight rollout: running phase sums plus the outcome and join facts for its recorded stub.

    ``sim_duration_s`` is the virtual time the rollout advanced (``n_steps * control_dt`` for a fixed-step
    loop), which becomes the recorded duration the reducer divides by wall time for the real-time factor.
    ``size_bytes`` is the real on-disk cost of one rollout's recording; left ``None``, the reducer measures
    the stub dir instead (near zero). ``success`` / ``terminated`` / ``scored`` become the ``eval.*`` verdicts
    the reducer reads for the success rate.
    """

    task: str
    trial: int
    uid: str
    sim_duration_s: float | None = None
    size_bytes: int | None = None
    success: bool | None = None
    terminated: bool | None = None
    scored: bool = False
    reset_s: float = 0.0
    env_step_s: float = 0.0
    policy_wait_s: float = 0.0
    record_io_s: float = 0.0
    env_physics_s: float = 0.0
    env_render_s: float = 0.0
    env_server_s: float = 0.0
    infer_ms: list[float] = field(default_factory=list)
    _wall_start: float = 0.0
    _discarded: bool = False

    def add_reset(self, seconds: float) -> None:
        self.reset_s += seconds

    def add_env_step(self, seconds: float) -> None:
        self.env_step_s += seconds

    def add_infer(self, seconds: float) -> None:
        """One policy round-trip: the whole of ``policy_wait_s`` and one entry in the latency distribution."""
        self.policy_wait_s += seconds
        self.infer_ms.append(seconds * 1000.0)

    def add_record_io(self, seconds: float) -> None:
        self.record_io_s += seconds

    def add_env_phases(self, physics_s: float, render_s: float, server_s: float) -> None:
        """One env step's server-reported decomposition: physics substeps, rendering, whole in-step wall."""
        self.env_physics_s += physics_s
        self.env_render_s += render_s
        self.env_server_s += server_s

    def reset(self) -> contextlib.AbstractContextManager[None]:
        return _timed(self.add_reset)

    def env_step(self) -> contextlib.AbstractContextManager[None]:
        return _timed(self.add_env_step)

    def policy(self) -> contextlib.AbstractContextManager[None]:
        return _timed(self.add_infer)

    def record_io(self) -> contextlib.AbstractContextManager[None]:
        return _timed(self.add_record_io)

    def set_outcome(self, *, success: bool | None = None, terminated: bool | None = None, scored: bool = False) -> None:
        """Record the eval verdicts that drive the success rate.

        A meaningful rate needs either ``success`` (taken as-is) or ``scored=True`` with ``terminated`` (a
        no-success termination then counts as a failure). Leaving all three unset keeps the episode out of the
        rate, and the run still reduces.
        """
        self.success = success
        self.terminated = terminated
        self.scored = scored

    def discard(self) -> None:
        """Drop this rollout without recording it — an abort that should not weigh on the pass."""
        self._discarded = True

    def to_timing(self, wall_s: float) -> EpisodeTiming:
        measured = self.reset_s + self.env_step_s + self.policy_wait_s + self.record_io_s
        return EpisodeTiming(
            task=self.task,
            trial=self.trial,
            episode_uid=self.uid,
            wall_s=wall_s,
            reset_s=self.reset_s,
            env_step_s=self.env_step_s,
            policy_wait_s=self.policy_wait_s,
            record_io_s=self.record_io_s,
            overhead_s=max(wall_s - measured, 0.0),
            infer_ms=[round(ms, 3) for ms in self.infer_ms],
            finished_at=time.time(),
            env_physics_s=self.env_physics_s,
            env_render_s=self.env_render_s,
            env_server_s=self.env_server_s,
        )

    def eval_static(self) -> dict[str, bool]:
        """The ``eval.*`` verdicts to write into ``static.json``, omitting the ones left unset."""
        static: dict[str, bool] = {}
        if self.success is not None:
            static['eval.success'] = self.success
        if self.terminated is not None:
            static['eval.terminated'] = self.terminated
        if self.scored:
            static['eval.scored'] = True
        return static


class TimingShim:
    """Collects one ``EpisodeTiming`` per rollout and writes the dataset dir the reducer consumes.

    The harness records episodes strictly in sequence — one is in flight at a time — so episode ids are a
    simple counter and the recorded-stub layout mirrors positronic's block/episode numbering.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        task: str | None = None,
        sample_gpu: bool = True,
        sample_host: bool = True,
        write_episodes: bool = True,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._task = task
        self._sample_gpu = sample_gpu
        self._sample_host = sample_host
        self._write_episodes = write_episodes
        self._records: list[EpisodeTiming] = []
        self._next_episode_id = 0
        self._sampler: subprocess.Popen | None = None
        self._host_sampler: _HostSampler | None = None

    @contextlib.contextmanager
    def run(self) -> Iterator['TimingShim']:
        """Bind the pass: create the output dir, start GPU + host sampling, and flush ``timing.jsonl`` on exit.

        A retry reusing the same ``--out`` dir must not mix a prior attempt's episode dirs into this
        pass — the reducer would join stale episodes as if they were fresh — so stale block dirs and
        ``timing.jsonl`` are cleared up front.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        for stale in self._out_dir.glob('[0-9]' * 12):
            shutil.rmtree(stale)
        (self._out_dir / TIMING_FILENAME).unlink(missing_ok=True)
        self._records.clear()
        self._next_episode_id = 0
        self.start_gpu()
        self.start_host()
        try:
            yield self
        finally:
            self.close()

    def start_gpu(self) -> None:
        if self._sample_gpu:
            self._sampler = start_gpu_sampler(self._out_dir)

    def start_host(self) -> None:
        """Start the host CPU/RAM sampler; a no-op when host sampling is off or ``/proc`` is unreadable."""
        if not self._sample_host:
            return
        sampler = _HostSampler(self._out_dir)
        if sampler.start():
            self._host_sampler = sampler

    def begin_episode(
        self,
        trial: int,
        task: str | None = None,
        *,
        uid: str | None = None,
        sim_duration_s: float | None = None,
        size_bytes: int | None = None,
    ) -> Episode:
        """Open a new in-flight rollout. ``uid`` defaults to a fresh id and is reused as the join key."""
        resolved_task = task if task is not None else self._task
        if resolved_task is None:
            raise ValueError('task is required: pass it to TimingShim(task=...) or episode(task=...)')
        episode = Episode(
            task=resolved_task,
            trial=trial,
            uid=uid if uid is not None else uuid.uuid4().hex,
            sim_duration_s=sim_duration_s,
            size_bytes=size_bytes,
        )
        episode._wall_start = time.perf_counter()
        return episode

    def finish_episode(self, episode: Episode) -> None:
        """Seal an in-flight rollout: write its recorded stub and append its timing record."""
        if episode._discarded:
            return
        wall_s = time.perf_counter() - episode._wall_start
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        if self._write_episodes:
            _write_episode_dir(self._out_dir, episode_id, episode)
        self._records.append(episode.to_timing(wall_s))

    @contextlib.contextmanager
    def episode(
        self,
        trial: int,
        task: str | None = None,
        *,
        uid: str | None = None,
        sim_duration_s: float | None = None,
        size_bytes: int | None = None,
    ) -> Iterator[Episode]:
        """Time one rollout end to end. A propagating exception aborts it (not recorded), like ``discard``."""
        episode = self.begin_episode(trial, task, uid=uid, sim_duration_s=sim_duration_s, size_bytes=size_bytes)
        try:
            yield episode
        except BaseException:
            episode.discard()
            raise
        finally:
            self.finish_episode(episode)

    def close(self) -> Path:
        """Stop GPU + host sampling and write every collected record as one JSON object per line; return the path."""
        self._stop_gpu()
        self._stop_host()
        path = self._out_dir / TIMING_FILENAME
        with path.open('w') as f:
            for record in self._records:
                f.write(json.dumps(asdict(record)) + '\n')
        logger.info(f'TimingShim: wrote {len(self._records)} episode timings to {path}')
        return path

    def _stop_gpu(self) -> None:
        if self._sampler is None:
            return
        self._sampler.terminate()
        try:
            self._sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._sampler.kill()
        self._sampler = None

    def _stop_host(self) -> None:
        if self._host_sampler is None:
            return
        self._host_sampler.stop()
        self._host_sampler = None


@contextlib.contextmanager
def _timed(sink: Callable[[float], None]) -> Iterator[None]:
    """Feed the wall duration of the enclosed block to ``sink``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink(time.perf_counter() - start)


def _write_episode_dir(root: Path, episode_id: int, episode: Episode) -> Path:
    """Write the recorded-episode stub the reducer joins against, in positronic's block/episode layout."""
    block_dir = root / f'{(episode_id // 1000) * 1000:012d}'
    ep_dir = block_dir / f'{episode_id:012d}'
    ep_dir.mkdir(parents=True, exist_ok=True)

    created_ts_ns = time.time_ns()
    duration_ns = int((episode.sim_duration_s or 0.0) * 1e9)
    meta: dict[str, object] = {'uid': episode.uid, 'created_ts_ns': created_ts_ns, 'duration_ns': duration_ns}
    if episode.size_bytes is not None:
        meta['size_mb'] = episode.size_bytes / (1024 * 1024)
    (ep_dir / 'meta.json').write_text(json.dumps(meta))
    (ep_dir / 'static.json').write_text(json.dumps(episode.eval_static()))
    _write_signal(ep_dir / SIGNAL_FILENAME, created_ts_ns, duration_ns)
    return ep_dir


def _write_signal(path: Path, start_ts_ns: int, duration_ns: int) -> None:
    """Write a two-sample timestamped signal so the episode's duration derives from data as well as meta."""
    end_ts_ns = start_ts_ns + max(duration_ns, 1)
    table = pa.table({
        'timestamp': pa.array([start_ts_ns, end_ts_ns], type=pa.int64()),
        'value': pa.array([0.0, 1.0], type=pa.float64()),
    })
    pq.write_table(table, path)


def _read_proc_stat() -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    """``((idle, total), [(idle, total), ...])`` cumulative jiffies for the whole CPU and each core.

    ``idle`` folds in ``iowait`` (the two idle columns); ``total`` sums every column. ``None`` when
    ``/proc/stat`` is unreadable — a box without procfs — so host sampling degrades to absent, not an error.
    """
    try:
        text = Path('/proc/stat').read_text()
    except OSError:
        return None
    overall: tuple[int, int] | None = None
    cores: list[tuple[int, int]] = []
    for line in text.splitlines():
        if not line.startswith('cpu'):
            break  # the cpu/cpuN lines lead /proc/stat; the rest (intr, ctxt, …) is irrelevant here
        parts = line.split()
        nums = [int(x) for x in parts[1:]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        pair = (idle, sum(nums))
        if parts[0] == 'cpu':
            overall = pair
        elif parts[0][3:].isdigit():
            cores.append(pair)
    if overall is None:
        return None
    return overall, cores


def _read_meminfo() -> tuple[int, int] | None:
    """``(mem_total_kb, mem_available_kb)`` from ``/proc/meminfo``, or ``None`` when unreadable."""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    total_kb: int | None = None
    avail_kb: int | None = None
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        if key == 'MemTotal':
            total_kb = int(rest.split()[0])
        elif key == 'MemAvailable':
            avail_kb = int(rest.split()[0])
        if total_kb is not None and avail_kb is not None:
            break
    if total_kb is None or avail_kb is None:
        return None
    return total_kb, avail_kb


def _cpu_busy_pct(prev: tuple[int, int], cur: tuple[int, int]) -> float:
    """Percent busy over the interval between two ``(idle, total)`` jiffie snapshots."""
    idle_delta = cur[0] - prev[0]
    total_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return 0.0
    return round(100.0 * (1.0 - idle_delta / total_delta), 2)


class _HostSampler:
    """Daemon thread logging per-interval host CPU (``/proc/stat``) and RAM (``/proc/meminfo``) samples.

    Writes ``host_stats.log`` — one JSON object per interval carrying monotonic + wall timestamps, overall and
    per-core CPU utilisation (busy fraction over that interval), and memory total/used/available in GiB. Works
    on any Linux box, GPU or CPU. A missing ``/proc`` at start yields no log (``start`` returns ``False``); a
    read that begins failing mid-pass simply stops emitting, so the pass is never taken down by its telemetry.
    """

    def __init__(self, out_dir: Path, *, interval_s: float = 1.0) -> None:
        self._log_path = out_dir / HOST_LOG_FILENAME
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='timingshim-host', daemon=True)
        self._baseline: tuple[tuple[int, int], list[tuple[int, int]]] | None = None
        self._fh = None

    def start(self) -> bool:
        """Capture the CPU baseline and launch the sampler thread; ``False`` when ``/proc/stat`` is unreadable."""
        self._baseline = _read_proc_stat()
        if self._baseline is None:
            logger.info('/proc/stat unavailable; skipping host CPU/RAM sampling')
            return False
        self._log_path.unlink(missing_ok=True)
        self._fh = self._log_path.open('w')
        self._thread.start()
        return True

    def _run(self) -> None:
        prev_overall, prev_cores = self._baseline
        # Wait one interval before the first sample so every logged line is a real delta off the baseline.
        while not self._stop.wait(self._interval_s):
            stat = _read_proc_stat()
            mem = _read_meminfo()
            if stat is None or mem is None:
                continue
            overall, cores = stat
            total_kb, avail_kb = mem
            sample = {
                't_mono': round(time.monotonic(), 3),
                't_wall': round(time.time(), 3),
                'cpu_pct': _cpu_busy_pct(prev_overall, overall),
                'per_core_pct': [_cpu_busy_pct(p, c) for p, c in zip(prev_cores, cores, strict=False)],
                'mem_total_gb': round(total_kb / (1024 * 1024), 3),
                'mem_used_gb': round((total_kb - avail_kb) / (1024 * 1024), 3),
                'mem_avail_gb': round(avail_kb / (1024 * 1024), 3),
            }
            self._fh.write(json.dumps(sample) + '\n')
            self._fh.flush()
            prev_overall, prev_cores = overall, cores

    def stop(self) -> None:
        """Signal the thread, join it, and close the log."""
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 5.0)
        if self._fh is not None:
            self._fh.close()
            self._fh = None
