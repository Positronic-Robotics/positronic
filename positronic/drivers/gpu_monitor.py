"""``GpuMonitor``: a foreground control system that samples the box's GPU and emits it as a signal.

Opt-in eval telemetry (``positronic eval run --timing``): a bundled ``timing.gpu`` sample per tick that the
recorder fans out to ``timing.gpu_util`` (whole-box utilisation %), ``timing.gpu_mem`` (whole-box memory,
MiB) and ``timing.gpu_mem_proc`` (the memory attributed to this eval's process tree, MiB). Per-process
*utilisation* is deliberately not attempted — it is unreliable under MPS / co-location — so only memory is
attributed.

``start`` spins a background daemon thread that does the blocking ``nvidia-smi`` reads at true wall cadence,
appending each probe to a buffer; it is started before the World runs so the thread is already sampling
during the harness's first synchronous reset. The cooperative ``run`` loop drains the whole buffer every
scheduler tick and emits the probes as one timestamped batch, so every probe is retained — a synchronous
span longer than the sampling interval (a long reset or env step) no longer collapses to a single reading.
Each probe carries its real capture wall-clock time as a ``timing.gpu_wall_ns`` value, so a reducer
reconstructs the true load-over-time from that. With no ``nvidia-smi`` on PATH the system is inert (a CPU
box): no thread starts, it emits nothing, and it keeps yielding so it never stops the eval.

A sim eval runs on a virtual clock that is frozen while the scheduler is blocked in a synchronous span, so a
batch's probes cannot each get a distinct virtual instant; they are placed at the drain instant with strictly
increasing timestamps (which the signal writer requires) while their real capture times ride in the
``timing.gpu_wall_ns`` values. The virtual placement is therefore coarse for a blocked span, but no probe and
no value is lost.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pimm
from positronic.dataset.serializers import Timestamped

logger = logging.getLogger(__name__)


@dataclass
class _GpuSample:
    """One box GPU reading: whole-box utilisation (%), whole-box memory used (MiB), the memory used by this
    eval's process tree (MiB), and the real wall-clock time it was captured (epoch ns)."""

    util_pct: float
    mem_mib: float
    proc_mem_mib: float
    wall_ns: int


def _run_nvidia_smi(args: list[str]) -> str | None:
    """``nvidia-smi`` stdout for ``args``, or ``None`` when the call fails (no binary, driver error, timeout)."""
    try:
        proc = subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout


def _process_tree_pids() -> set[int]:
    """This process and all its descendants — the env server subprocess and its Isaac children are spawned
    under the harness process, so the tree rooted here is exactly this eval's processes."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as stat_file:
                # After the ``(comm)`` field (comm may hold spaces/parens): state, ppid, ...
                ppid = int(stat_file.read().rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    tree = {os.getpid()}
    stack = [os.getpid()]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree


def _read_gpu_sample(device: str) -> _GpuSample | None:
    """One sample for ``device``: whole-box util+memory, plus this eval's process-tree memory. ``None`` when
    the whole-box query fails (no GPU / driver error) or reports a metric as non-numeric — ``N/A`` /
    ``[Not Supported]``, e.g. on some MIG configurations — so the caller records nothing this sample and the
    daemon thread keeps sampling instead of dying on the conversion."""
    box = _run_nvidia_smi(['--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits', '-i', device])
    if box is None:
        return None
    util_s, mem_s = (part.strip() for part in box.strip().splitlines()[0].split(','))
    try:
        util_pct, mem_mib = float(util_s), float(mem_s)
    except ValueError:
        return None

    pids = _process_tree_pids()
    proc_mib = 0.0
    apps = _run_nvidia_smi(['--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits', '-i', device])
    for line in (apps or '').strip().splitlines():
        if not line.strip():
            continue
        pid_s, used_s = (part.strip() for part in line.split(','))
        try:
            if int(pid_s) in pids:
                proc_mib += float(used_s)  # ``[Not Supported]`` / permission strings raise and are skipped
        except ValueError:
            continue
    return _GpuSample(util_pct=util_pct, mem_mib=mem_mib, proc_mem_mib=proc_mib, wall_ns=time.time_ns())


class GpuMonitor(pimm.ControlSystem):
    """Samples the box's GPU and emits the probes buffered since the last tick as one batch of ``timing.gpu``
    samples the recorder fans out to ``timing.gpu_*`` (``_util``, ``_mem``, ``_mem_proc``, ``_wall_ns``).
    ``sampling_hz`` is the wall cadence of the underlying ``nvidia-smi`` reads."""

    def __init__(self, sampling_hz: float = 1.0):
        self._interval = 1.0 / sampling_hz
        # Sample only the GPU this eval runs on — the first CUDA-visible device, else device 0. Left unpinned,
        # nvidia-smi reports every visible GPU and idle/unrelated devices would dilute the numbers.
        self._device = (os.environ.get('CUDA_VISIBLE_DEVICES', '') or '0').split(',')[0]
        self._lock = threading.Lock()
        self._buffer: list[_GpuSample] = []
        self._last_emit_ts = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.output = pimm.ControlSystemEmitter(self)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            sample = _read_gpu_sample(self._device)
            if sample is not None:
                with self._lock:
                    self._buffer.append(sample)
            self._stop.wait(self._interval)

    def start(self) -> None:
        """Spin the wall-cadence sampling thread. Started before the World runs so it is already sampling
        during the harness's first synchronous reset. Inert (no thread) without ``nvidia-smi`` on PATH."""
        if self._thread is not None:
            return
        if shutil.which('nvidia-smi') is None:
            logger.info('GpuMonitor: no nvidia-smi on PATH; GPU telemetry disabled')
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread and join it. Safe to call when never started."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> 'GpuMonitor':
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _drain_batch(self, now_ns: int) -> list[Timestamped]:
        """Every probe buffered since the last tick as one timestamped batch, retaining all of them (a
        blocked span's probes are not collapsed to the latest). The virtual clock is frozen during a blocked
        span, so the probes share one instant; they are placed at ``now_ns`` with strictly increasing ts (what
        the signal writer requires) while each carries its real capture wall-ns as ``timing.gpu_wall_ns`` data,
        from which a reducer recovers the true load-over-time regardless of the coarse virtual placement."""
        with self._lock:
            batch, self._buffer = self._buffer, []
        samples = []
        for probe in batch:
            ts = max(now_ns, self._last_emit_ts + 1)
            self._last_emit_ts = ts
            value = {
                '_util': probe.util_pct,
                '_mem': probe.mem_mib,
                '_mem_proc': probe.proc_mem_mib,
                '_wall_ns': probe.wall_ns,
            }
            samples.append(Timestamped(ts, value))
        return samples

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Yield every scheduler round (not a Sleep(interval)): the wall cadence lives in the sampling thread,
        # so the loop drains every probe buffered since the last tick and emits them as one timestamped batch.
        # Probes captured while the scheduler was blocked in a synchronous span are all retained, not collapsed
        # to the latest. The sim simulator sleeps each step and paces the clock, so this Yield is non-pacing.
        while not should_stop.value:
            batch = self._drain_batch(clock.now_ns())
            if batch:
                self.output.emit(batch, clock.now_ns())
            yield pimm.Yield()
