"""Wall-clock telemetry sidecars for ``positronic eval run --timing``.

A sim eval runs on a virtual clock, so the virtual time a rollout advances says nothing about the real
compute it cost. This module captures that operational signal — nested spans for the wall-clock phase split
(reset, env step, inference, record IO) and a free-running machine-load sampler (CPU, memory, GPU) — as
sidecar files next to the recorded dataset, never mixed into it. The dataset records the robot's world; these
files describe the machinery around it, in wall time.

Storage is one set of files per process under ``<out_dir>/telemetry/``: ``<process>.meta.json`` (process
identity), ``<process>.spans.jsonl`` (OTLP/JSON-lines spans), ``<process>.stats.jsonl`` (one machine-load
sample per line). The env server writes its own set; nothing rides over the wire.

Instrumented code sees one seam: ``from positronic import telemetry`` then ``with telemetry.span('reset'):``.
The span helpers no-op while unbound (a normal eval binds nothing), so a call site carries no ``None`` check.
The pass-level report is an offline reduce over the raw files (``positronic.cli.eval.timing_report``).
"""

import functools
import json
import logging
import os
import platform
import socket
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import pynvml
from opentelemetry import trace
from opentelemetry.exporter.otlp.json.file import FileSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

from positronic.simulator.env_server.telemetry import SPAN_ENV_RESET as SPAN_ENV_RESET
from positronic.simulator.env_server.telemetry import SPAN_ENV_STEP as SPAN_ENV_STEP

logger = logging.getLogger(__name__)

_SCOPE = 'positronic'

# Span names and attribute keys are the producer↔consumer contract: a producer opens a span (or stamps an
# attribute) by these literals, and the offline reduce (``positronic.cli.eval.timing_report``) matches on the
# same ones. Owned here so both sides import one name. ``env.step``/``env.reset`` are owned by the stdlib-only
# env-server writer (the isolated env interpreter cannot import this module) and re-exported above.
SPAN_EVAL_PASS = 'eval.pass'
SPAN_EPISODE = 'episode'
SPAN_RESET = 'reset'
SPAN_MATERIALIZE = 'materialize'
SPAN_POLICY_INFER = 'policy.infer'
SPAN_RECORD_IO = 'record.io'

ATTR_EPISODE_INDEX = 'episode.index'
ATTR_EPISODE_STEPS = 'episode.steps'
ATTR_EPISODE_VIRTUAL_S = 'episode.virtual_s'
ATTR_EPISODE_ABORTED = 'episode.aborted'
ATTR_EPISODE_PARTIAL = 'episode.partial'
ATTR_PASS_FAILED = 'pass.failed'

# The harness process's sidecar name — the discriminator between client-side spans (episode, client env.step)
# and an env server's own file, which reduces rely on.
HARNESS_PROCESS = 'harness'

# The subdirectory under a run's output dir where every process's sidecars live. The eval CLI writer, a
# launched env server (via the env-var it is handed) and the offline reduce all resolve to this same dir.
TELEMETRY_SUBDIR = 'telemetry'

# The bound provider and the current pass/episode spans are process-global: the harness, env proxy and
# recorder run as cooperative control systems in one thread and interleave their spans, so parenting is
# resolved here rather than through OTel's ambient context (which does not survive the scheduler's generator
# hops). ``span`` reads these to parent a per-tick span to the episode in flight; a nested span (materialize
# inside env.step) still parents to whatever OTel span is current within its own ``with`` block.
_provider: TracerProvider | None = None
_current_pass: Span | None = None
_current_episode: Span | None = None


# The unbound fallback is a PRIVATE no-op, never ``trace.get_tracer``: a host application embedding this code
# may have configured OTel's global provider, and an untimed run must not export spans through it.
_NOOP_TRACER = trace.NoOpTracer()

# NVML's uint64 "value not available" sentinel. On a driver that cannot attribute a process's GPU memory, the
# query succeeds and pynvml surfaces this raw value in ``usedGpuMemory`` rather than ``None``.
_NVML_VALUE_NOT_AVAILABLE = 2**64 - 1


def _tracer() -> trace.Tracer:
    """The bound provider's tracer while a run is timed, else a private no-op tracer."""
    return _provider.get_tracer(_SCOPE) if _provider is not None else _NOOP_TRACER


def _attr_value(value: Any) -> Any:
    """One span-attribute value coerced to what OTel accepts: a flat scalar or a homogeneous scalar list.
    Anything nested is JSON-encoded, so a call site can pass a trial-context value without a shape check."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (bool, int, float, str)) for item in value):
        return list(value)
    return json.dumps(value)


def _encode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {key: _attr_value(value) for key, value in attrs.items()}


@contextmanager
def bind(out_dir: Path | str, process: str, run_id: str) -> Generator[TracerProvider, None, None]:
    """Provider lifecycle for one process's telemetry: write ``<process>.meta.json``, stream spans to
    ``<process>.spans.jsonl``, and register the provider so ``span`` records. The batch processor is flushed
    and shut down on exit — an abrupt exit would otherwise lose its queued tail."""
    global _provider
    telemetry_dir = Path(out_dir) / TELEMETRY_SUBDIR
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    meta = {
        'schema': 1,
        'run_id': run_id,
        'host': host,
        'process': process,
        'pid': os.getpid(),
        'started_wall_ns': time.time_ns(),
        'started_mono_ns': time.monotonic_ns(),
        'python': platform.python_version(),
        'platform': platform.platform(),
    }
    (telemetry_dir / f'{process}.meta.json').write_text(json.dumps(meta, indent=2))

    resource = Resource.create({'run.id': run_id, 'process.name': process, 'host.name': host})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(telemetry_dir / f'{process}.spans.jsonl')))
    _provider = provider
    try:
        yield provider
    finally:
        provider.force_flush()
        provider.shutdown()
        _provider = None


def force_flush() -> None:
    """Flush the batch processor's queue so a crash after this point loses no already-ended span; called at
    each episode end so a mid-pass crash loses at most one episode's tail. Inert while unbound."""
    if _provider is not None:
        _provider.force_flush()


def _per_tick_parent() -> Any:
    """The parent context for a per-tick span. A currently-active span of THIS provider's trace means a
    genuinely nested span (materialize inside env.step) — ``None`` lets it parent ambiently. Otherwise the
    span parents to the episode in flight (else the pass): between scheduler hops nothing of ours is current,
    and a host application's unrelated current span must not adopt telemetry spans — they would leave the
    trace the report reduces. With no pass or episode open there is nothing to anchor to; ambient stands."""
    anchor = _current_episode if _current_episode is not None else _current_pass
    if anchor is None:
        return None
    current = trace.get_current_span().get_span_context()
    if current.is_valid and current.trace_id == anchor.get_span_context().trace_id:
        return None
    return trace.set_span_in_context(anchor)


def span(name: str, **attrs: Any):
    """A wall-clock span named ``name``, entered as the current span for the enclosed block. A per-tick span
    (no OTel span currently active) parents to the episode in flight; a span opened inside another parents to
    it. A no-op while unbound."""
    return _tracer().start_as_current_span(name, context=_per_tick_parent(), attributes=_encode_attrs(attrs))


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
    like a per-tick ``span``. A no-op while unbound."""
    recorded = _tracer().start_span(
        name, context=_per_tick_parent(), start_time=start_ns, attributes=_encode_attrs(attrs)
    )
    recorded.end(end_time=end_ns)


@contextmanager
def eval_pass(**attrs: Any) -> Generator[Span, None, None]:
    """The pass span bracketing a whole eval sweep; per-episode spans parent to it. Held as a module span
    rather than the OTel-current one so it does not become the parent of the per-tick spans. A sweep that
    exits with an exception still exports its pass — the partial window is real recorded data — stamped
    ``pass.failed`` so a reduce can see (and report) that it did not run to completion."""
    global _current_pass
    pass_span = _tracer().start_span(SPAN_EVAL_PASS, attributes=_encode_attrs(attrs))
    _current_pass = pass_span
    try:
        yield pass_span
    except BaseException:
        pass_span.set_attribute(ATTR_PASS_FAILED, True)
        raise
    finally:
        pass_span.end()
        _current_pass = None


def begin_episode(**attrs: Any) -> Span:
    """Open the episode span (parented to the pass) and register it as the parent for this episode's per-tick
    spans. A no-op while unbound returns an invalid span the caller ends harmlessly."""
    global _current_episode
    context = trace.set_span_in_context(_current_pass) if _current_pass is not None else None
    episode = _tracer().start_span(SPAN_EPISODE, context=context, attributes=_encode_attrs(attrs))
    _current_episode = episode
    return episode


def end_episode(episode: Span, **attrs: Any) -> None:
    """Stamp the episode's end attributes (steps, virtual duration), end the span, and flush."""
    global _current_episode
    if attrs:
        episode.set_attributes(_encode_attrs(attrs))
    episode.end()
    if _current_episode is episode:
        _current_episode = None
    force_flush()


def discard_episode(episode: Span) -> None:
    """End an aborted episode's span, marked ``episode.aborted`` so the reduce can drop it."""
    global _current_episode
    episode.set_attribute(ATTR_EPISODE_ABORTED, True)
    episode.end()
    if _current_episode is episode:
        _current_episode = None


def end_partial_episode(episode: Span, **attrs: Any) -> None:
    """End an episode span left open by a failure mid-rollout (a raising ``reset`` / ``new_session`` / session
    call), stamping its end attributes (steps, virtual duration), marking it ``episode.partial``, and flushing.
    Ending it is what exports it: the batch processor never emits an unended span, so an abandoned episode's
    finished children would orphan and the reduce would lose their phases. Marked (not aborted) so the reduce
    keeps it — its finished phases attribute — while flagging that it did not run to completion."""
    global _current_episode
    if attrs:
        episode.set_attributes(_encode_attrs(attrs))
    episode.set_attribute(ATTR_EPISODE_PARTIAL, True)
    episode.end()
    if _current_episode is episode:
        _current_episode = None
    force_flush()


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
                process = str(resource_attrs.get('process.name', ''))
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
                'i': index,
                'util_pct': float(util.gpu),
                'mem_used_b': int(memory.used),
                'mem_total_b': int(memory.total),
                'power_w': self._power_w(handle),
                'proc_mem_b': self._process_memory(handle, tree_pids),
                # Per-process GPU utilisation is not reliably attributable under MPS / co-location, so it is
                # left unmeasured (device util above is real); per-process memory is attributed.
                'proc_util_pct': None,
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
            cached = self._proc_cache.get(proc.pid)
            if cached is None:
                cached = proc
                self._proc_cache[proc.pid] = cached
                cached.cpu_percent(interval=None)  # prime; the first read is meaningless
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
            't_ns': time.time_ns(),
            'cpu_sys_pct': psutil.cpu_percent(interval=None),
            'iowait_pct': float(getattr(cpu_times, 'iowait', 0.0)),
            'mem_sys_used_b': int(psutil.virtual_memory().used),
            'cpu_proc_pct': cpu_proc_pct,
            'rss_proc_b': rss_proc_b,
            'gpu_count': self._nvml.device_count,
            'gpus': self._nvml.sample(tree_pids),
        }

    def _loop(self) -> None:
        _seal_truncated_line(self._path)
        with open(self._path, 'a') as file:
            while not self._stop.is_set():
                file.write(json.dumps(self._sample(), separators=(',', ':')) + '\n')
                file.flush()
                self._stop.wait(self._interval)
