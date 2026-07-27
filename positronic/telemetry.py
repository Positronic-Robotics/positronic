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

import json
import logging
import os
import platform
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import pynvml
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_SCOPE = 'positronic'

# The bound provider and the current pass/episode spans are process-global: the harness, env proxy and
# recorder run as cooperative control systems in one thread and interleave their spans, so parenting is
# resolved here rather than through OTel's ambient context (which does not survive the scheduler's generator
# hops). ``span`` reads these to parent a per-tick span to the episode in flight; a nested span (materialize
# inside env.step) still parents to whatever OTel span is current within its own ``with`` block.
_provider: TracerProvider | None = None
_current_pass: Span | None = None
_current_episode: Span | None = None


def _tracer() -> trace.Tracer:
    """The bound provider's tracer while a run is timed, else the API default (a no-op tracer)."""
    return _provider.get_tracer(_SCOPE) if _provider is not None else trace.get_tracer(_SCOPE)


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
def bind(out_dir: Path | str, process: str, run_id: str) -> Iterator[TracerProvider]:
    """Provider lifecycle for one process's telemetry: write ``<process>.meta.json``, stream spans to
    ``<process>.spans.jsonl``, and register the provider so ``span`` records. The batch processor is flushed
    and shut down on exit — an abrupt exit would otherwise lose its queued tail."""
    global _provider
    telemetry_dir = Path(out_dir) / 'telemetry'
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
    provider.add_span_processor(BatchSpanProcessor(_FileSpanExporter(telemetry_dir / f'{process}.spans.jsonl')))
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


def span(name: str, **attrs: Any):
    """A wall-clock span named ``name``, entered as the current span for the enclosed block. A per-tick span
    (no OTel span currently active) parents to the episode in flight; a span opened inside another parents to
    it. A no-op while unbound."""
    context = None
    current = trace.get_current_span()
    if not current.get_span_context().is_valid:
        parent = _current_episode if _current_episode is not None else _current_pass
        if parent is not None:
            context = trace.set_span_in_context(parent)
    return _tracer().start_as_current_span(name, context=context, attributes=_encode_attrs(attrs))


@contextmanager
def eval_pass(**attrs: Any) -> Iterator[Span]:
    """The pass span bracketing a whole eval sweep; per-episode spans parent to it. Held as a module span
    rather than the OTel-current one so it does not become the parent of the per-tick spans."""
    global _current_pass
    pass_span = _tracer().start_span('eval.pass', attributes=_encode_attrs(attrs))
    _current_pass = pass_span
    try:
        yield pass_span
    finally:
        pass_span.end()
        _current_pass = None


def begin_episode(**attrs: Any) -> Span:
    """Open the episode span (parented to the pass) and register it as the parent for this episode's per-tick
    spans. A no-op while unbound returns an invalid span the caller ends harmlessly."""
    global _current_episode
    context = trace.set_span_in_context(_current_pass) if _current_pass is not None else None
    episode = _tracer().start_span('episode', context=context, attributes=_encode_attrs(attrs))
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
    episode.set_attribute('episode.aborted', True)
    episode.end()
    if _current_episode is episode:
        _current_episode = None


def _otlp_value(value: Any) -> dict[str, Any]:
    """One attribute value as an OTLP/JSON ``AnyValue``. ``bool`` is checked before ``int`` (it subclasses
    ``int``), and OTLP carries integers as strings."""
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        return {'intValue': str(value)}
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, str):
        return {'stringValue': value}
    if isinstance(value, (list, tuple)):
        return {'arrayValue': {'values': [_otlp_value(item) for item in value]}}
    return {'stringValue': json.dumps(value)}


def _otlp_attributes(attrs: Any) -> list[dict[str, Any]]:
    return [{'key': key, 'value': _otlp_value(value)} for key, value in attrs.items()]


def _encode_span(span_data: ReadableSpan) -> dict[str, Any]:
    context = span_data.context
    encoded: dict[str, Any] = {
        'traceId': f'{context.trace_id:032x}' if context is not None else '',
        'spanId': f'{context.span_id:016x}' if context is not None else '',
        'name': span_data.name,
        'startTimeUnixNano': str(span_data.start_time or 0),
        'endTimeUnixNano': str(span_data.end_time or 0),
        'attributes': _otlp_attributes(span_data.attributes or {}),
    }
    if span_data.parent is not None:
        encoded['parentSpanId'] = f'{span_data.parent.span_id:016x}'
    return encoded


class _FileSpanExporter(SpanExporter):
    """Writes each exported batch as one OTLP/JSON line to ``<process>.spans.jsonl``, flushed per line so a
    crash loses at most the batch in flight. The reader parses the same shape back."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._file = open(path, 'a')

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        resource_attrs = _otlp_attributes(spans[0].resource.attributes)
        doc = {
            'resourceSpans': [
                {
                    'resource': {'attributes': resource_attrs},
                    'scopeSpans': [{'scope': {'name': _SCOPE}, 'spans': [_encode_span(s) for s in spans]}],
                }
            ]
        }
        with self._lock:
            self._file.write(json.dumps(doc, separators=(',', ':')) + '\n')
            self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            self._file.close()


class SpanRec(NamedTuple):
    """One parsed span: hex ``span_id``/``parent_id`` (``parent_id`` is ``None`` for a root), wall-clock
    epoch-ns bounds, and the flat attribute map."""

    name: str
    start_ns: int
    end_ns: int
    attrs: dict[str, Any]
    span_id: str
    parent_id: str | None


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
                for scope_spans in resource_spans.get('scopeSpans', []):
                    for span_data in scope_spans.get('spans', []):
                        yield SpanRec(
                            name=span_data['name'],
                            start_ns=int(span_data['startTimeUnixNano']),
                            end_ns=int(span_data['endTimeUnixNano']),
                            attrs=_decode_attrs(span_data.get('attributes', [])),
                            span_id=span_data['spanId'],
                            parent_id=span_data.get('parentSpanId') or None,
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
        self._ok = False
        try:
            pynvml.nvmlInit()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._ok = True
        except pynvml.NVMLError as error:
            logger.info('telemetry: NVML unavailable (%s); GPU stats disabled', error)

    def sample(self, tree_pids: set[int]) -> list[dict[str, Any]]:
        gpus: list[dict[str, Any]] = []
        for index, handle in enumerate(self._handles):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                'i': index,
                'util_pct': float(util.gpu),
                'mem_used_b': int(memory.used),
                'mem_total_b': int(memory.total),
                'power_w': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                'proc_mem_b': self._process_memory(handle, tree_pids),
                # Per-process GPU utilisation is not reliably attributable under MPS / co-location, so it is
                # left unmeasured (device util above is real); per-process memory is attributed.
                'proc_util_pct': None,
            })
        return gpus

    def _process_memory(self, handle: Any, tree_pids: set[int]) -> int | None:
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError as error:
            if not self._proc_mem_warned:
                logger.info('telemetry: per-process GPU memory unavailable (%s); proc_mem_b left null', error)
                self._proc_mem_warned = True
            return None
        used = 0
        for proc in procs:
            if proc.pid in tree_pids and proc.usedGpuMemory is not None:
                used += int(proc.usedGpuMemory)
        return used

    def shutdown(self) -> None:
        if self._ok:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


class StatsSampler:
    """A daemon thread sampling host CPU/memory, this process tree's CPU/memory, and per-GPU load at ``hz``,
    writing one JSON line per sample to ``out_path`` (flushed per line, so a crash keeps every prior sample).
    It free-runs on wall time — no span context, no episode boundary — so no sample is lost at a phase edge."""

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
            'gpus': self._nvml.sample(tree_pids),
        }

    def _loop(self) -> None:
        with open(self._path, 'a') as file:
            while not self._stop.is_set():
                file.write(json.dumps(self._sample(), separators=(',', ':')) + '\n')
                file.flush()
                self._stop.wait(self._interval)
