"""Wall-clock telemetry sidecars for ``positronic eval run --timing``.

A sim eval runs on a virtual clock, so the virtual time a rollout advances says nothing about the real
compute it cost. This module captures that operational signal — nested wall-clock spans and a free-running
machine-load sampler (CPU, memory, GPU) — as sidecar files next to the recorded dataset, never mixed into it.
The dataset records the robot's world; these files describe the machinery around it, in wall time.

The mechanism is domain-blind: it knows spans, anchors and files, and nothing about episodes, passes or
inference. A long-running span's lifecycle belongs to whoever owns that phase — the harness opens and closes
the rollout's span, the eval CLI the pass's.

That splits the contract literals by **who writes the bytes they name**. This module owns the names of what it
writes itself: the machine-load sample's fields, the sidecar suffixes, the telemetry subdirectory. The span
names and attribute keys are written by eval-domain code THROUGH the span helpers, which pass them opaquely
and never match on them, so they live in ``positronic.telemetry_keys`` — holding them here would mean knowing
what an episode is.

Storage is one set of files per process under ``<out_dir>/telemetry/``: ``<process>.spans.jsonl``
(OTLP/JSON-lines spans, whose resource block carries the process identity) and ``<process>.stats.jsonl`` (one
machine-load sample per line). The env server writes its own set; nothing rides over the wire.

Instrumented code sees one seam: ``from positronic import telemetry`` then ``with telemetry.span('reset'):``.
The span helpers no-op while unbound (a normal eval binds nothing), so a call site carries no ``None`` check.
The pass-level report is an offline reduce over the raw files (``positronic.cli.eval.timing_report``).

An instrumented call site needs only the OTel API, a default dependency, for the no-op span surface. The
``telemetry`` extra adds the OTel SDK and pynvml, imported by ``bind`` and ``StatsSampler``, the two entry
points ``--timing`` reaches.
"""

import functools
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span

# The resource-block attribute keys are defined by the stdlib-only env-server writer — the isolated env
# interpreter cannot import positronic, so the constrained side owns the names and this one follows them.
from positronic.simulator.env_server.telemetry import (
    ATTR_HOST_NAME,
    ATTR_PROCESS_NAME,
    ATTR_PROCESS_PID,
    ATTR_RUN_ID,
    SPANS_SUFFIX,
)

if TYPE_CHECKING:
    import psutil
    import pynvml
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

_SCOPE = 'positronic'

_MISSING_EXTRA = (
    '--timing needs the telemetry extra: `uv sync --extra telemetry` (or `pip install positronic[telemetry]`)'
)

# The subdirectory under a run's output dir where every process's sidecars live. The eval CLI writer, a
# launched env server (via the env-var it is handed) and the offline reduce all resolve to this same dir.
TELEMETRY_SUBDIR = 'telemetry'

# The machine-load sample's field names, and the suffix of the file holding one such sample per line. Owned
# here because ``StatsSampler`` writes them; the reduce, their only reader, imports them.
STATS_SUFFIX = '.stats.jsonl'
STAT_T_NS = 't_ns'
STAT_CPU_SYS_PCT = 'cpu_sys_pct'
STAT_IOWAIT_PCT = 'iowait_pct'
STAT_MEM_SYS_USED_B = 'mem_sys_used_b'
STAT_CPU_PROC_PCT = 'cpu_proc_pct'
STAT_RSS_PROC_B = 'rss_proc_b'
STAT_GPU_COUNT = 'gpu_count'
STAT_GPUS = 'gpus'

# The per-device fields of one ``STAT_GPUS`` entry.
GPU_INDEX = 'i'
GPU_UTIL_PCT = 'util_pct'
GPU_MEM_USED_B = 'mem_used_b'
GPU_MEM_TOTAL_B = 'mem_total_b'
GPU_POWER_W = 'power_w'
GPU_PROC_MEM_B = 'proc_mem_b'
GPU_PROC_UTIL_PCT = 'proc_util_pct'

# The bound provider is process-global; the anchor stack is per-context. An anchor is a long-running span its
# owner holds open (a pass, a rollout).
#
# - A span opened while no span of the bound trace is active parents to the innermost anchor.
# - A span opened inside another parents to that one.
# - With nothing anchored, a span roots.
# - A copied context carries the anchors it was copied with; a push or pop after the copy does not reach it.
# - The stack stands in for OTel's ambient context, which does not survive the scheduler's generator hops.
_provider: 'TracerProvider | None' = None
_anchors: ContextVar[tuple[Span, ...]] = ContextVar('positronic_telemetry_anchors', default=())


# The unbound fallback is a PRIVATE no-op, never ``trace.get_tracer``: a host application embedding this code
# may have configured OTel's global provider, and an untimed run must not export spans through it.
_NOOP_TRACER = trace.NoOpTracer()

# NVML's uint64 "value not available" sentinel. On a driver that cannot attribute a process's GPU memory, the
# query succeeds and pynvml surfaces this raw value in ``usedGpuMemory`` rather than ``None``.
_NVML_VALUE_NOT_AVAILABLE = 2**64 - 1


def _tracer() -> trace.Tracer:
    """The bound provider's tracer while a run is timed, else a private no-op tracer."""
    return _provider.get_tracer(_SCOPE) if _provider is not None else _NOOP_TRACER


def spans_path(out_dir: Path | str, process: str) -> Path:
    """Where ``process`` writes its span sidecar under a run's output dir."""
    return Path(out_dir) / TELEMETRY_SUBDIR / f'{process}{SPANS_SUFFIX}'


def stats_path(out_dir: Path | str, process: str) -> Path:
    """Where ``process`` writes its machine-load sidecar under a run's output dir."""
    return Path(out_dir) / TELEMETRY_SUBDIR / f'{process}{STATS_SUFFIX}'


def _attr_value(value: Any) -> Any:
    """One span-attribute value coerced to what OTel accepts: a flat scalar or a homogeneous scalar list.
    Anything nested is JSON-encoded, so a call site can pass a trial param value without a shape check."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        # OTel drops an array whose elements disagree in type, so a mixed sequence goes the JSON route: the
        # value survives as text rather than the attribute vanishing from the span.
        scalars = all(isinstance(item, (bool, int, float, str)) for item in items)
        if scalars and len({type(item) for item in items}) <= 1:
            return items
    return json.dumps(value)


def _encode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {key: _attr_value(value) for key, value in attrs.items()}


@contextmanager
def bind(out_dir: Path | str, process: str, run_id: str) -> Generator['TracerProvider', None, None]:
    """Provider lifecycle for one process's telemetry: stream spans to ``<process>.spans.jsonl`` under a
    resource block carrying this process's identity, and register the provider so ``span`` records. The batch
    processor is flushed and shut down on exit — an abrupt exit would otherwise lose its queued tail."""
    global _provider
    try:  # the OTel SDK and its file exporter ship in the optional `telemetry` extra
        from opentelemetry.exporter.otlp.json.file import FileSpanExporter  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(_MISSING_EXTRA) from error
    path = spans_path(out_dir, process)
    path.parent.mkdir(parents=True, exist_ok=True)
    resource = Resource.create({
        ATTR_RUN_ID: run_id,
        ATTR_PROCESS_NAME: process,
        ATTR_PROCESS_PID: os.getpid(),
        ATTR_HOST_NAME: socket.gethostname(),
    })
    # Sample every span. Left to its default the SDK reads OTEL_TRACES_SAMPLER, so a host application's
    # `always_off` or ratio setting would silence this provider too — the sidecar comes back empty or partial
    # and the report reads a timed run as untimed. What this provider records goes to its own file and nowhere
    # else, so there is nothing for a sampling budget to protect.
    provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)

    # A killed predecessor can leave a truncated final line; seal it so the appending exporter starts a fresh
    # line — otherwise read_spans merges the first new record into the fragment and skips both.
    _seal_truncated_line(path)
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(path)))
    _provider = provider
    try:
        yield provider
    finally:
        provider.force_flush()
        provider.shutdown()
        _provider = None


def force_flush() -> None:
    """Flush the batch processor's queue so a crash after this point loses no already-ended span; the owner of
    a long-running span flushes as it closes, so a later crash loses at most that span's tail. Inert while
    unbound."""
    if _provider is not None:
        _provider.force_flush()


def push_anchor(anchor: Span) -> None:
    """Make ``anchor`` the innermost anchor: spans opened outside it from now on parent to it."""
    _anchors.set((*_anchors.get(), anchor))


def pop_anchor(anchor: Span) -> None:
    """Drop ``anchor`` from the stack, wherever in it it sits — a failure path can close anchors out of order,
    and a stale one would go on parenting later spans into a finished span's trace."""
    _anchors.set(tuple(held for held in _anchors.get() if held is not anchor))


def _anchor_context() -> Any:
    """The context parenting a span to the innermost anchor, or a root context when nothing is anchored.

    Rooting is explicit rather than ``None``: a host application's current span would adopt the span into a
    foreign trace, and under parent-based sampling an unsampled one silences the whole sidecar.
    """
    anchors = _anchors.get()
    return trace.set_span_in_context(anchors[-1]) if anchors else Context()


def _anchor_parent() -> Any:
    """The parent context for a span opened outside an active one: ``None`` where the current span belongs to
    this provider's trace, so a nested span parents ambiently."""
    anchors = _anchors.get()
    if anchors:
        current = trace.get_current_span().get_span_context()
        if current.is_valid and current.trace_id == anchors[-1].get_span_context().trace_id:
            return None
    return _anchor_context()


def start_span(name: str, **attrs: Any) -> Span:
    """Start a span its caller holds and ends itself, without entering it as the OTel-current span. A no-op
    while unbound returns an invalid span the caller ends harmlessly."""
    return _tracer().start_span(name, context=_anchor_context(), attributes=_encode_attrs(attrs))


def set_attrs(span: Span, **attrs: Any) -> None:
    """Stamp attributes on a span the caller holds, coerced to what OTel accepts."""
    span.set_attributes(_encode_attrs(attrs))


def span(name: str, **attrs: Any):
    """A wall-clock span named ``name``, entered as the current span for the enclosed block. A no-op while
    unbound."""
    return _tracer().start_as_current_span(name, context=_anchor_parent(), attributes=_encode_attrs(attrs))


def traced(name: str, **attrs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of ``span`` for a function whose whole body is one span: run the call inside a span
    named ``name``. A no-op while unbound, like ``span``."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name, **attrs):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def record_span(name: str, start_ns: int, end_ns: int, **attrs: Any) -> None:
    """Record an already-elapsed span with explicit wall-clock bounds — for a phase whose emit is decided only
    after it ran (a policy round-trip that turned out to be a real inference, not a scheduler replay). Parents
    like ``span``. A no-op while unbound."""
    recorded = _tracer().start_span(
        name, context=_anchor_parent(), start_time=start_ns, attributes=_encode_attrs(attrs)
    )
    recorded.end(end_time=end_ns)


def _seal_truncated_line(path: Path) -> None:
    """Seal a predecessor's truncated final line (a killed run can die mid-write) with a newline before
    appending, so the first new record does not merge into the fragment and get skipped along with it."""
    try:
        with open(path, 'rb') as file:
            file.seek(-1, os.SEEK_END)
            sealed = file.read(1) == b'\n'
    except (FileNotFoundError, OSError):
        return  # absent or empty: nothing to seal
    if not sealed:
        with open(path, 'ab') as file:
            file.write(b'\n')


class SpanRec(NamedTuple):
    """One parsed span: hex ``span_id``/``parent_id`` (``parent_id`` is ``None`` for a root), wall-clock
    epoch-ns bounds, the flat attribute map, and the recording process's name (the ``process.name`` resource
    attribute every sidecar writer stamps — ``''`` when a file carries none)."""

    name: str
    start_ns: int
    end_ns: int
    attrs: dict[str, Any]
    span_id: str
    parent_id: str | None
    process: str = ''


def _decode_value(value: dict[str, Any]) -> Any:
    if 'stringValue' in value:
        return value['stringValue']
    if 'boolValue' in value:
        return value['boolValue']
    if 'intValue' in value:
        return int(value['intValue'])
    if 'doubleValue' in value:
        return value['doubleValue']
    if 'arrayValue' in value:
        return [_decode_value(item) for item in value['arrayValue'].get('values', [])]
    return None


def _decode_attrs(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    return {attr['key']: _decode_value(attr['value']) for attr in attributes}


def read_spans(path: Path | str) -> Iterator[SpanRec]:
    """Parse the OTLP/JSON-lines spans file, yielding each span. A truncated final line (crash mid-write) is
    tolerated: an unparseable line is skipped rather than raising."""
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            for resource_spans in doc.get('resourceSpans', []):
                resource_attrs = _decode_attrs(resource_spans.get('resource', {}).get('attributes', []))
                process = str(resource_attrs.get(ATTR_PROCESS_NAME, ''))
                for scope_spans in resource_spans.get('scopeSpans', []):
                    for span_data in scope_spans.get('spans', []):
                        yield SpanRec(
                            name=span_data['name'],
                            start_ns=int(span_data['startTimeUnixNano']),
                            end_ns=int(span_data['endTimeUnixNano']),
                            attrs=_decode_attrs(span_data.get('attributes', [])),
                            span_id=span_data['spanId'],
                            parent_id=span_data.get('parentSpanId') or None,
                            process=process,
                        )


def read_stats(path: Path | str) -> Iterator[dict[str, Any]]:
    """Parse the machine-load stats file, one sample per line. A truncated final line is skipped."""
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class _Nvml:
    """NVML device handles, initialised once. Degrades to no GPUs when the driver/library is absent, and to
    device-only fields when per-process memory is unavailable (a PID namespace without ``--pid=host``); each
    degradation logs once."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._proc_mem_warned = False
        self._power_warned = False
        self._device_warned: set[int] = set()
        self._ok = False
        try:
            pynvml.nvmlInit()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._ok = True
        except pynvml.NVMLError as error:
            logger.info('telemetry: NVML unavailable (%s); GPU stats disabled', error)

    @property
    def device_count(self) -> int:
        """The configured GPU count (the number of NVML device handles), authoritative even when a device is
        omitted from a sample after a mid-run query error. ``0`` when NVML is unavailable."""
        return len(self._handles)

    def sample(self, tree_pids: set[int]) -> list[dict[str, Any]]:
        gpus: list[dict[str, Any]] = []
        for index, handle in enumerate(self._handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except pynvml.NVMLError as error:
                # A device can refuse a query outright (MIG) or drop mid-run (transiently lost). Skip it and
                # keep sampling — one device's failure must not kill the whole stats stream.
                if index not in self._device_warned:
                    logger.info('telemetry: GPU %d stats unavailable (%s); device skipped', index, error)
                    self._device_warned.add(index)
                continue
            gpus.append({
                GPU_INDEX: index,
                GPU_UTIL_PCT: float(util.gpu),
                GPU_MEM_USED_B: int(memory.used),
                GPU_MEM_TOTAL_B: int(memory.total),
                GPU_POWER_W: self._power_w(handle),
                GPU_PROC_MEM_B: self._process_memory(handle, tree_pids),
                # Per-process GPU utilisation is not reliably attributable under MPS / co-location, so it is
                # left unmeasured (device util above is real); per-process memory is attributed.
                GPU_PROC_UTIL_PCT: None,
            })
        return gpus

    def _power_w(self, handle: Any) -> float | None:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError as error:
            # Not every GPU supports the power query; a dead optional field must not kill the sampler thread.
            if not self._power_warned:
                logger.info('telemetry: GPU power unavailable (%s); power_w left null', error)
                self._power_warned = True
            return None

    def _process_memory(self, handle: Any, tree_pids: set[int]) -> int | None:
        # Compute AND graphics contexts: a renderer (Isaac's Vulkan/GL) holds its memory as a graphics
        # context, so the compute list alone misses the sim's biggest GPU consumer.
        try:
            procs = [
                *pynvml.nvmlDeviceGetComputeRunningProcesses(handle),
                *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle),
            ]
        except pynvml.NVMLError as error:
            if not self._proc_mem_warned:
                logger.info('telemetry: per-process GPU memory unavailable (%s); proc_mem_b left null', error)
                self._proc_mem_warned = True
            return None
        # NVML reports host pids. In a PID namespace without ``--pid=host`` the tree pids are namespace-local,
        # so every membership test below misses and the device reads a confident 0 — a GPU-heavy eval published
        # as using no VRAM. The host mapping is unreadable from inside the namespace, so translating is out;
        # but a reported pid that resolves to no process here is that mismatch. Every reported pid must resolve:
        # a namespace-local pid can collide with an unrelated host pid, and one such collision is enough to make
        # the whole set look comparable and charge a stranger's memory to this eval. On the host they all
        # resolve, including other tenants', so a device this eval does not touch still reads 0.
        if procs and not all(psutil.pid_exists(proc.pid) for proc in procs):
            if not self._proc_mem_warned:
                logger.info('telemetry: GPU processes live in another PID namespace; proc_mem_b left null')
                self._proc_mem_warned = True
            return None
        # A process with both context types (Isaac: CUDA + Vulkan) appears in BOTH lists, and drivers commonly
        # report the process's total in each — merge by pid with max, which never double-counts.
        used_by_pid: dict[int, int] = {}
        for proc in procs:
            if proc.pid not in tree_pids or proc.usedGpuMemory is None:
                continue
            # A driver that cannot attribute this process's memory returns NVML's uint64 sentinel (~18 EiB), not
            # ``None``. Summing it would publish garbage, so treat it as unattributable: return ``None`` for the
            # whole device — the same signal as an NVMLError — so the reduce counts the sample as incomplete.
            if proc.usedGpuMemory == _NVML_VALUE_NOT_AVAILABLE:
                if not self._proc_mem_warned:
                    logger.info('telemetry: per-process GPU memory unavailable (NVML sentinel); proc_mem_b left null')
                    self._proc_mem_warned = True
                return None
            used_by_pid[proc.pid] = max(used_by_pid.get(proc.pid, 0), int(proc.usedGpuMemory))
        return sum(used_by_pid.values())

    def shutdown(self) -> None:
        if self._ok:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


class StatsSampler:
    """A daemon thread sampling host CPU/memory, this process tree's CPU/memory, and per-GPU load at ``hz``,
    writing one JSON line per sample to ``out_path`` (flushed per line, so a crash keeps every prior sample).
    It free-runs on wall time — no span context, no episode boundary — so no sample is lost at a phase edge.
    It runs as an in-process daemon thread, not a subprocess: ``os.getpid()`` gives exact process-tree
    attribution and the lifecycle stays trivial, at the cost of going blind if the harness itself hard-hangs —
    load-of-a-hang forensics this profiling does not need."""

    def __init__(self, out_path: Path | str, hz: float = 1.0) -> None:
        global psutil, pynvml
        try:  # the sampler's probes ship in the optional `telemetry` extra
            import psutil  # noqa: PLC0415
            import pynvml  # noqa: PLC0415
        except ImportError as error:
            raise RuntimeError(_MISSING_EXTRA) from error
        self._path = Path(out_path)
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='telemetry-stats', daemon=True)
        self._nvml = _Nvml()
        self._self = psutil.Process(os.getpid())
        self._proc_cache: dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # prime the host CPU counter so the first sample is a real delta

    def __enter__(self) -> 'StatsSampler':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._nvml.shutdown()

    def _tree(self) -> tuple[set[int], float, int]:
        """This process tree's pids, summed CPU percent, and summed RSS. Process objects are cached across
        samples so ``cpu_percent`` reads a real interval delta; dead processes are dropped."""
        try:
            processes = [self._self, *self._self.children(recursive=True)]
        except psutil.Error:
            processes = [self._self]
        pids = {proc.pid for proc in processes}
        for pid in list(self._proc_cache):
            if pid not in pids:
                del self._proc_cache[pid]
        cpu_pct = 0.0
        rss = 0
        for proc in processes:
            if proc.pid not in self._proc_cache:
                self._proc_cache[proc.pid] = proc
                proc.cpu_percent(interval=None)  # prime; the first read is meaningless
            cached = self._proc_cache[proc.pid]
            try:
                cpu_pct += cached.cpu_percent(interval=None)
                rss += cached.memory_info().rss
            except psutil.Error:
                self._proc_cache.pop(proc.pid, None)
        return pids, cpu_pct, rss

    def _sample(self) -> dict[str, Any]:
        cpu_times = psutil.cpu_times_percent(interval=None)
        tree_pids, cpu_proc_pct, rss_proc_b = self._tree()
        return {
            STAT_T_NS: time.time_ns(),
            STAT_CPU_SYS_PCT: psutil.cpu_percent(interval=None),
            STAT_IOWAIT_PCT: float(getattr(cpu_times, 'iowait', 0.0)),
            STAT_MEM_SYS_USED_B: int(psutil.virtual_memory().used),
            STAT_CPU_PROC_PCT: cpu_proc_pct,
            STAT_RSS_PROC_B: rss_proc_b,
            STAT_GPU_COUNT: self._nvml.device_count,
            STAT_GPUS: self._nvml.sample(tree_pids),
        }

    def _loop(self) -> None:
        _seal_truncated_line(self._path)
        with open(self._path, 'a') as file:
            while not self._stop.is_set():
                file.write(json.dumps(self._sample(), separators=(',', ':')) + '\n')
                file.flush()
                self._stop.wait(self._interval)
