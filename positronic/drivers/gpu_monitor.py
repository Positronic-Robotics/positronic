"""``GpuMonitor``: a foreground control system that samples the box's GPU and emits it as a signal.

Opt-in eval telemetry (``positronic eval run --timing``): a bundled ``timing.gpu`` sample per tick that the
recorder fans out to ``timing.gpu_util`` (whole-box utilisation %), ``timing.gpu_mem`` (whole-box memory,
MiB) and ``timing.gpu_mem_proc`` (the memory attributed to this eval's process tree, MiB). Per-process
*utilisation* is deliberately not attempted — it is unreliable under MPS / co-location — so only memory is
attributed.

A background daemon thread does the blocking ``nvidia-smi`` reads at true wall cadence into a shared latest
sample; the cooperative ``run`` loop emits that latest sample each tick stamped by the world clock. A sim
eval runs on a virtual clock, so an async sample has no wall->virtual mapping — sampling foreground and
stamping with the world clock is what keeps the sample inside the episode bounds (the recorder also records
the real epoch in ``extra_ts['system']``). With no ``nvidia-smi`` on PATH the system is inert (a CPU box):
it emits nothing and keeps yielding so it never stops the eval.
"""

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import pimm

logger = logging.getLogger(__name__)


@dataclass
class _GpuSample:
    """One box GPU reading: whole-box utilisation (%), whole-box memory used (MiB), and the memory used by
    this eval's process tree (MiB)."""

    util_pct: float
    mem_mib: float
    proc_mem_mib: float


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
    the whole-box query fails (no GPU / driver error), so the caller records nothing this sample."""
    box = _run_nvidia_smi(['--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits', '-i', device])
    if box is None:
        return None
    util_s, mem_s = (part.strip() for part in box.strip().splitlines()[0].split(','))

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
    return _GpuSample(util_pct=float(util_s), mem_mib=float(mem_s), proc_mem_mib=proc_mib)


class GpuMonitor(pimm.ControlSystem):
    """Samples the box's GPU and emits a bundled ``timing.gpu`` sample the recorder fans out to
    ``timing.gpu_*``. ``sampling_hz`` is the wall cadence of the underlying ``nvidia-smi`` reads."""

    def __init__(self, sampling_hz: float = 1.0):
        self._interval = 1.0 / sampling_hz
        # Sample only the GPU this eval runs on — the first CUDA-visible device, else device 0. Left unpinned,
        # nvidia-smi reports every visible GPU and idle/unrelated devices would dilute the numbers.
        self._device = (os.environ.get('CUDA_VISIBLE_DEVICES', '') or '0').split(',')[0]
        self._lock = threading.Lock()
        self._latest: _GpuSample | None = None
        self._seq = 0
        self.output = pimm.ControlSystemEmitter(self)

    def _sample_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            sample = _read_gpu_sample(self._device)
            if sample is not None:
                with self._lock:
                    self._latest = sample
                    self._seq += 1
            stop.wait(self._interval)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        if shutil.which('nvidia-smi') is None:
            logger.info('GpuMonitor: no nvidia-smi on PATH; GPU telemetry disabled')
            while not should_stop.value:
                yield pimm.Sleep(self._interval)
            return

        stop = threading.Event()
        thread = threading.Thread(target=self._sample_loop, args=(stop,), daemon=True)
        thread.start()
        emitted_seq = 0
        try:
            while not should_stop.value:
                with self._lock:
                    seq, sample = self._seq, self._latest
                if sample is not None and seq != emitted_seq:
                    emitted_seq = seq
                    self.output.emit(
                        {'_util': sample.util_pct, '_mem': sample.mem_mib, '_mem_proc': sample.proc_mem_mib},
                        clock.now_ns(),
                    )
                yield pimm.Sleep(self._interval)
        finally:
            stop.set()
            thread.join(timeout=5)
