import json
import os
import sys
import time
from contextlib import contextmanager

import psutil
import pynvml
import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from positronic import telemetry
from positronic.simulator.env_server.telemetry import ENV_PROCESS
from positronic.telemetry_keys import HARNESS_PROCESS, SPAN_EVAL_PASS


def _spans_by_name(path):
    return {rec.name: rec for rec in telemetry.read_spans(path)}


@contextmanager
def _anchored(name, **attrs):
    """One anchor's lifetime, the shape every anchor owner (the eval CLI's pass, the harness's episode) drives
    by hand: start a span the caller holds, anchor it, end and unanchor it."""
    span = telemetry.start_span(name, **attrs)
    telemetry.push_anchor(span)
    try:
        yield span
    finally:
        span.end()
        telemetry.pop_anchor(span)


@pytest.fixture
def without_telemetry_extra(monkeypatch):
    """A default install: the OTel API is there, the ``telemetry`` extra's packages are not. ``None`` in
    ``sys.modules`` is what the import system raises ImportError on."""
    for name in (
        'psutil',
        'pynvml',
        'opentelemetry.sdk',
        'opentelemetry.sdk.resources',
        'opentelemetry.sdk.trace',
        'opentelemetry.sdk.trace.export',
        'opentelemetry.exporter.otlp.json.file',
    ):
        monkeypatch.setitem(sys.modules, name, None)


def test_span_surface_survives_without_the_telemetry_extra(without_telemetry_extra):
    """Every instrumented call site sits on the OTel API alone, so an install carrying no telemetry extra runs
    a normal eval — the helpers no-op exactly as they do while unbound."""
    with _anchored('outer'):
        with telemetry.span('probe'):
            pass


def test_recording_without_the_telemetry_extra_names_the_extra(tmp_path, without_telemetry_extra):
    """Both entry points ``--timing`` reaches fail with the install command, not an ImportError on a package
    name the operator has no reason to connect to ``--timing``."""
    with pytest.raises(RuntimeError, match='telemetry extra'):
        with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-no-extra'):
            pass
    with pytest.raises(RuntimeError, match='telemetry extra'):
        telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')


def test_nested_spans_round_trip(tmp_path):
    """A bound run's spans parse back with anchor parenting — an unnested span under the innermost anchor, an
    anchor under the anchor enclosing it, one opened inside another under that one — plus hex ids and decoded
    attributes."""
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-xyz'):
        with _anchored('outer', policy='stub'):
            with _anchored('inner', index=0) as inner:
                with telemetry.span('probe'):
                    pass
                with telemetry.span('phase'):
                    with telemetry.span('sub-phase'):
                        pass
                telemetry.set_attrs(inner, steps=5, virtual_s=1.5)

    spans_path = telemetry.spans_path(tmp_path, HARNESS_PROCESS)
    spans = _spans_by_name(spans_path)
    assert set(spans) == {'outer', 'inner', 'probe', 'phase', 'sub-phase'}

    # Hex ids: 16 nibbles for a span id, and every id parses as hex.
    for rec in spans.values():
        assert len(rec.span_id) == 16
        int(rec.span_id, 16)

    assert spans['outer'].parent_id is None
    assert spans['inner'].parent_id == spans['outer'].span_id
    assert spans['probe'].parent_id == spans['inner'].span_id
    assert spans['phase'].parent_id == spans['inner'].span_id
    assert spans['sub-phase'].parent_id == spans['phase'].span_id

    assert spans['outer'].attrs['policy'] == 'stub'
    assert spans['inner'].attrs['steps'] == 5
    assert spans['inner'].attrs['virtual_s'] == 1.5
    assert spans['inner'].end_ns >= spans['inner'].start_ns


def test_mixed_type_attribute_sequence_survives_as_json(tmp_path):
    """OTel drops an attribute array whose elements disagree in type, so a mixed sequence — a trial context
    holding a label beside a number — is JSON-encoded rather than lost between the caller and the sidecar."""
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-mixed'):
        with telemetry.span('probe', mixed=[1, 'two'], plain=[1, 2]):
            pass

    attrs = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))['probe'].attrs
    assert json.loads(attrs['mixed']) == [1, 'two']
    assert attrs['plain'] == [1, 2]  # a single-type sequence still travels as an array


def test_anchor_popped_out_of_order_stops_parenting(tmp_path):
    """An owner closing an anchor that is no longer the innermost — a failure path unwinding several at once —
    drops it from wherever it sits, so no finished span goes on adopting later spans."""
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-unwind'):
        outer = telemetry.start_span('outer')
        telemetry.push_anchor(outer)
        inner = telemetry.start_span('inner')
        telemetry.push_anchor(inner)

        telemetry.pop_anchor(outer)  # the enclosing anchor closes first
        outer.end()
        with telemetry.span('probe'):
            pass
        inner.end()
        telemetry.pop_anchor(inner)
        with telemetry.span('after'):
            pass

    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert spans['probe'].parent_id == spans['inner'].span_id  # the innermost anchor still stands
    assert spans['after'].parent_id is None  # nothing anchored: the span roots rather than adopting a corpse


def test_unbound_span_is_inert(tmp_path):
    """Off ``--timing`` nothing binds: the span helpers no-op, write no file, and raise nothing."""
    with _anchored('outer'):
        with telemetry.span('probe'):
            pass
    assert not (tmp_path / 'telemetry').exists()


def test_resource_carries_process_identity(tmp_path):
    """Every span document's resource block names the run and the writing process, so a sidecar identifies
    itself without a second file."""
    with telemetry.bind(tmp_path, ENV_PROCESS, 'run-1'):
        with telemetry.span('probe'):
            pass

    line = json.loads((telemetry.spans_path(tmp_path, ENV_PROCESS)).read_text().splitlines()[0])
    attrs = telemetry._decode_attrs(line['resourceSpans'][0]['resource']['attributes'])
    assert attrs[telemetry.ATTR_RUN_ID] == 'run-1'
    assert attrs[telemetry.ATTR_PROCESS_NAME] == ENV_PROCESS
    assert attrs[telemetry.ATTR_PROCESS_PID] == os.getpid()


class _FakeUtil:
    gpu = 42


class _FakeMem:
    used = 3 * 1024**3
    total = 8 * 1024**3


class _FakeProc:
    def __init__(self, pid, used):
        self.pid = pid
        self.usedGpuMemory = used


def test_unbound_span_never_touches_global_tracer(monkeypatch):
    """Unbound telemetry is a PRIVATE no-op: a host application may have configured OTel's global tracer
    provider, and an untimed run must not export spans (or their attributes) through it."""

    def _boom(*args, **kwargs):
        raise AssertionError('unbound telemetry must not consult the global tracer provider')

    monkeypatch.setattr(telemetry.trace, 'get_tracer', _boom)
    with telemetry.span('probe', index=0):
        pass


def test_anchored_span_ignores_foreign_ambient_span(tmp_path):
    """A host application's current OTel span (a different provider, a different trace) must not adopt our
    spans opened under an anchor: they parent to it, so the report still sees them."""
    foreign = SdkTracerProvider()
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-foreign'):
        with _anchored('outer'):
            with foreign.get_tracer('host-app').start_as_current_span('host-span'):
                with telemetry.span('probe'):
                    pass

    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert 'host-span' not in spans  # the foreign span belongs to the host's provider, not our file
    assert spans['probe'].parent_id == spans['outer'].span_id


def test_root_span_detaches_from_a_foreign_ambient_span(tmp_path):
    """With nothing anchored, a host application's current OTel span must not become the parent of ours.
    Inheriting it hands our trace the host's sampling decision: under parent-based sampling an unsampled host
    span makes the pass non-recording, and every episode and phase anchored beneath it is dropped — an empty
    sidecar, and a report that reads a timed run as untimed."""
    foreign = SdkTracerProvider(sampler=ALWAYS_OFF)
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-foreign-root'):
        with foreign.get_tracer('host-app').start_as_current_span('host-span'):
            with _anchored(SPAN_EVAL_PASS):
                with telemetry.span('probe'):
                    pass

    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert spans[SPAN_EVAL_PASS].parent_id is None  # a root of our own trace, not a child of the host's
    assert spans['probe'].parent_id == spans[SPAN_EVAL_PASS].span_id


def test_spans_survive_a_host_sampler_in_the_environment(tmp_path, monkeypatch):
    """OTEL_TRACES_SAMPLER configures the SDK's default sampler, so a host application that turns tracing off
    would take this provider's spans with it — an empty sidecar on a run that asked to be timed."""
    monkeypatch.setenv('OTEL_TRACES_SAMPLER', 'always_off')
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-sampler'):
        with _anchored(SPAN_EVAL_PASS):
            with telemetry.span('probe'):
                pass

    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert set(spans) == {'eval.pass', 'probe'}


def _install_fake_nvml(monkeypatch):
    monkeypatch.setattr(pynvml, 'nvmlInit', lambda: None)
    monkeypatch.setattr(pynvml, 'nvmlShutdown', lambda: None)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetCount', lambda: 1)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetHandleByIndex', lambda i: f'handle-{i}')
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetUtilizationRates', lambda h: _FakeUtil())
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetMemoryInfo', lambda h: _FakeMem())
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetPowerUsage', lambda h: 150_000)
    monkeypatch.setattr(
        pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [_FakeProc(os.getpid(), 1234 * 1024**2)]
    )
    # A renderer (Isaac's Vulkan/GL) holds its memory as a graphics context — attributed alongside compute.
    monkeypatch.setattr(
        pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [_FakeProc(os.getpid(), 100 * 1024**2)]
    )


def test_stats_sample_with_fake_gpu(tmp_path, monkeypatch):
    _install_fake_nvml(monkeypatch)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    sample = sampler._sample()
    assert set(sample) >= {
        telemetry.STAT_T_NS,
        telemetry.STAT_CPU_SYS_PCT,
        telemetry.STAT_IOWAIT_PCT,
        telemetry.STAT_MEM_SYS_USED_B,
        telemetry.STAT_CPU_PROC_PCT,
        telemetry.STAT_RSS_PROC_B,
        telemetry.STAT_GPUS,
    }
    assert len(sample[telemetry.STAT_GPUS]) == 1
    gpu = sample[telemetry.STAT_GPUS][0]
    assert gpu[telemetry.GPU_INDEX] == 0
    assert gpu[telemetry.GPU_UTIL_PCT] == 42.0
    assert gpu[telemetry.GPU_MEM_USED_B] == 3 * 1024**3
    assert gpu[telemetry.GPU_POWER_W] == 150.0
    # The same pid in the compute and graphics lists merges by max — drivers commonly report the process
    # total in each, so summing both entries would double-count.
    assert gpu[telemetry.GPU_PROC_MEM_B] == 1234 * 1024**2
    assert gpu[telemetry.GPU_PROC_UTIL_PCT] is None
    sampler._nvml.shutdown()


def test_stats_sample_treats_nvml_sentinel_as_unavailable(tmp_path, monkeypatch):
    """A driver that cannot attribute a process's GPU memory returns NVML's uint64 sentinel (~18 EiB), not
    ``None`` — recording it verbatim would publish garbage as the eval's peak process VRAM. The device reads
    unavailable (``proc_mem_b`` None) instead."""
    _install_fake_nvml(monkeypatch)
    sentinel_proc = _FakeProc(os.getpid(), telemetry._NVML_VALUE_NOT_AVAILABLE)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [sentinel_proc])
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [])
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    sample = sampler._sample()
    gpu = sample[telemetry.STAT_GPUS][0]
    assert gpu[telemetry.GPU_PROC_MEM_B] is None  # taken verbatim the sentinel reads as ~1.8e19 bytes of process VRAM
    sampler._nvml.shutdown()


def test_stats_sample_treats_foreign_pid_namespace_as_unavailable(tmp_path, monkeypatch):
    """NVML reports host pids; in a container without ``--pid=host`` the sampler's own tree pids are
    namespace-local, so no reported process matches and the device would read a confident 0 — a GPU-heavy eval
    published as using no VRAM. Pids that exist in no namespace we can see mark that mismatch, and the device
    reads unavailable instead."""
    _install_fake_nvml(monkeypatch)
    # A host pid the sampler's namespace cannot see, standing in for NVML reporting from outside the container.
    foreign = _FakeProc(4_000_000, 8 * 1024**3)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [foreign])
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [])
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: pid != foreign.pid)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    assert (
        sampler._sample()[telemetry.STAT_GPUS][0][telemetry.GPU_PROC_MEM_B] is None
    )  # a 0 here would be indistinguishable from idle
    sampler._nvml.shutdown()


def test_stats_sample_treats_a_partly_resolving_pid_set_as_unavailable(tmp_path, monkeypatch):
    """A namespace-local pid can collide with an unrelated host pid, so one reported process resolving here is
    no evidence the two pid spaces are the same. Attribution needs every reported pid to resolve; short of
    that the device reads unavailable rather than charging a stranger's memory to this eval."""
    _install_fake_nvml(monkeypatch)
    collision = _FakeProc(os.getpid(), 2 * 1024**3)  # a host pid that happens to name a live process here
    foreign = _FakeProc(4_000_000, 8 * 1024**3)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [collision, foreign])
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [])
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: pid != foreign.pid)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    assert sampler._sample()[telemetry.STAT_GPUS][0][telemetry.GPU_PROC_MEM_B] is None
    sampler._nvml.shutdown()


def test_stats_sample_reports_zero_when_gpu_is_idle(tmp_path, monkeypatch):
    """A device with no processes on it is genuinely 0 for this eval, not unavailable — the namespace check
    must not swallow the ordinary idle case."""
    _install_fake_nvml(monkeypatch)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [])
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [])
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    assert sampler._sample()[telemetry.STAT_GPUS][0][telemetry.GPU_PROC_MEM_B] == 0
    sampler._nvml.shutdown()


def test_stats_sample_skips_failing_device(tmp_path, monkeypatch):
    """A per-device NVML failure (a MIG device refusing utilisation, a transiently lost GPU) skips that device
    but keeps the sample — the stats stream must degrade, never die."""
    _install_fake_nvml(monkeypatch)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetCount', lambda: 2)

    def _util(handle):
        if handle == 'handle-0':
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
        return _FakeUtil()

    monkeypatch.setattr(pynvml, 'nvmlDeviceGetUtilizationRates', _util)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    sample = sampler._sample()
    assert [gpu[telemetry.GPU_INDEX] for gpu in sample[telemetry.STAT_GPUS]] == [1]
    assert isinstance(sample[telemetry.STAT_CPU_SYS_PCT], float)
    sampler._nvml.shutdown()


def test_stats_sample_without_gpu(tmp_path, monkeypatch):
    def _raise():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(pynvml, 'nvmlInit', _raise)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    sample = sampler._sample()
    assert sample[telemetry.STAT_GPUS] == []
    assert isinstance(sample[telemetry.STAT_CPU_SYS_PCT], float)
    assert isinstance(sample[telemetry.STAT_RSS_PROC_B], int)


def test_stats_sampler_thread_writes_lines(tmp_path, monkeypatch):
    def _raise():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(pynvml, 'nvmlInit', _raise)
    path = tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}'
    with telemetry.StatsSampler(path, hz=200.0):
        time.sleep(0.1)
    samples = list(telemetry.read_stats(path))
    assert samples, 'the sampler thread wrote at least one line'
    assert all(telemetry.STAT_T_NS in sample for sample in samples)


def test_bind_seals_truncated_predecessor_line(tmp_path):
    """A killed run can leave ``<process>.spans.jsonl`` ending in a truncated fragment with no trailing newline.
    The next bind into the same directory seals that line before its exporter appends, so the first new record
    starts a fresh line instead of concatenating onto the fragment and being skipped along with it."""
    spans_path = telemetry.spans_path(tmp_path, HARNESS_PROCESS)

    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-1'):
        with telemetry.span('first'):
            pass
    with open(spans_path, 'a') as file:
        file.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"nam')  # crash mid-write, no trailing newline

    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-2'):
        with telemetry.span('second'):
            pass

    names = [rec.name for rec in telemetry.read_spans(spans_path)]
    assert 'second' in names  # an unsealed fragment swallows the record appended after it
    assert 'first' in names  # the pre-existing valid span still reads back


def test_readers_tolerate_truncated_final_line(tmp_path):
    spans_path = telemetry.spans_path(tmp_path, HARNESS_PROCESS)
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-trunc'):
        with telemetry.span('probe'):
            pass
    with open(spans_path, 'a') as file:
        file.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"nam')  # crash mid-write
    names = [rec.name for rec in telemetry.read_spans(spans_path)]
    assert names == ['probe']

    stats_path = tmp_path / 'stats.jsonl'
    stats_path.write_text('{"t_ns": 1, "gpus": []}\n{"t_ns": 2, "cpu_sy')
    stats = list(telemetry.read_stats(stats_path))
    assert [sample[telemetry.STAT_T_NS] for sample in stats] == [1]
