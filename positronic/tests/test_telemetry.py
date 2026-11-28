import contextvars
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import psutil
import pynvml
import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

import pimm
from positronic import telemetry
from positronic.simulator.env_server.telemetry import ENV_PROCESS, ENV_RUN_ID, ENV_TELEMETRY_DIR
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


def test_scoped_spans_nest_without_an_episode_anchor(tmp_path):
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'scoped'):
        with telemetry.span('parent'):
            with telemetry.span('child'):
                pass
        with telemetry.span('outside'):
            pass
    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert spans['child'].parent_id == spans['parent'].span_id
    assert spans['outside'].parent_id is None


def test_timing_sink_without_exporter_reports_failures_and_restores_scope(monkeypatch):
    now = 0
    monkeypatch.setattr(telemetry.time, 'time_ns', lambda: now)
    timings = []
    with telemetry.timings_to(lambda *args: timings.append(args)):
        with pytest.raises(ValueError, match='failed'):
            with telemetry.span('outer'):
                with telemetry.span('inner'):
                    now = 10
                    raise ValueError('failed')
    with telemetry.span('outside'):
        pass
    assert timings == [('inner', 0, 10), ('outer', 0, 10)]


def test_a_span_recorded_with_explicit_bounds_reaches_the_timing_sink():
    timings = []
    with telemetry.timings_to(lambda *args: timings.append(args)):
        telemetry.record_span('wire.send', 3, 7)
    assert timings == [('wire.send', 3, 7)]


@pytest.mark.parametrize('anchored', [False, True])
def test_an_anchored_recorded_span_skips_the_active_span_for_the_anchor(tmp_path, anchored):
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'anchored-record'):
        with _anchored('episode'):
            with telemetry.span('round'):
                record = telemetry.record_anchored_span if anchored else telemetry.record_span
                record('hold', time.time_ns(), time.time_ns())
    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    parent = spans['episode'] if anchored else spans['round']
    assert spans['hold'].parent_id == parent.span_id


def test_mixed_type_attribute_sequence_survives_as_json(tmp_path):
    """OTel drops an attribute array whose elements disagree in type, so a mixed sequence — a trial param
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


def test_a_thread_records_under_the_anchor_it_was_dispatched_under(tmp_path):
    """The anchor stack is per-context, so work handed to a thread under a copy of it records where it was
    dispatched: a call outliving the anchor that asked for it still parents to that anchor, and the one
    anchored in its place does not adopt it."""
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'run-dispatch'), _anchored('pass'):
        asked = telemetry.start_span('asked')
        telemetry.push_anchor(asked)
        proceed = threading.Event()

        def late():
            proceed.wait(5)
            with telemetry.span('probe'):
                pass

        with ThreadPoolExecutor(max_workers=1) as pool:
            call = pool.submit(contextvars.copy_context().run, late)
            asked.end()
            telemetry.pop_anchor(asked)
            with _anchored('next'):
                proceed.set()
                call.result(timeout=5)

    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert spans['probe'].parent_id == spans['asked'].span_id


def test_unbound_span_is_inert(tmp_path):
    """Off ``--timing`` nothing binds: the span helpers no-op, write no file, and raise nothing."""
    with _anchored('outer'):
        with telemetry.span('probe'):
            pass
    assert not (tmp_path / 'telemetry').exists()


def _resource_attrs(path):
    """The resource block a sidecar stamps on every span, from its first line."""
    line = json.loads(path.read_text().splitlines()[0])
    return telemetry._decode_attrs(line['resourceSpans'][0]['resource']['attributes'])


def test_resource_carries_process_identity(tmp_path):
    """Every span document's resource block names the run and the writing process, so a sidecar identifies
    itself without a second file."""
    with telemetry.bind(tmp_path, ENV_PROCESS, 'run-1'):
        with telemetry.span('probe'):
            pass

    attrs = _resource_attrs(telemetry.spans_path(tmp_path, ENV_PROCESS))
    assert attrs[telemetry.ATTR_RUN_ID] == 'run-1'
    assert attrs[telemetry.ATTR_PROCESS_NAME] == ENV_PROCESS
    assert attrs[telemetry.ATTR_PROCESS_PID] == os.getpid()


def test_an_interrupted_world_flushes_telemetry_before_its_exception_is_released(tmp_path):
    closed = []
    owner = threading.get_ident()
    failure = RuntimeError('sibling failed')

    def recording(should_stop, clock):
        with telemetry.bind(tmp_path, HARNESS_PROCESS, 'interrupted'):
            with telemetry.span('completed'):
                pass
            try:
                while not should_stop.value:
                    yield pimm.Sleep(0.01)
            finally:
                closed.append(threading.get_ident())

    class Recording(pimm.ControlSystem):
        def run(self, should_stop, clock):
            return recording_loop

    class Failing(pimm.ControlSystem):
        def run(self, should_stop, clock):
            yield pimm.Sleep(0.01)
            raise failure

    with pimm.World(virtual_time=True) as world:
        recording_loop = recording(world.should_stop_reader(), world.clock)
        try:
            with pytest.raises(RuntimeError) as raised:
                world.run([Recording(), Failing()])
            assert raised.value is failure
            assert closed == [owner]
            assert not telemetry.enabled()
            telemetry.force_flush()
            spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, HARNESS_PROCESS)))
            assert [span.name for span in spans] == ['completed']
            with telemetry.bind(tmp_path, HARNESS_PROCESS, 'next'):
                with telemetry.span('next-run'):
                    pass
            assert not telemetry.enabled()
            spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, HARNESS_PROCESS)))
            assert [span.name for span in spans] == ['completed', 'next-run']
        finally:
            recording_loop.close()


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


def _refuse_nvml_utilisation(monkeypatch):
    """NVML as a Jetson's: it opens the GPU and refuses every utilisation and memory query."""
    _install_fake_nvml(monkeypatch)

    def _refuse(handle):
        raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    monkeypatch.setattr(pynvml, 'nvmlDeviceGetUtilizationRates', _refuse)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetMemoryInfo', _refuse)


def test_a_jetson_gpu_sample_takes_its_load_from_sysfs_when_nvml_refuses(tmp_path, monkeypatch):
    _refuse_nvml_utilisation(monkeypatch)
    load = tmp_path / 'load'
    load.write_text('204\n')
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}', tegra_gpu_load_path=load)
    sample = sampler._sample()
    sampler._nvml.shutdown()
    assert sample[telemetry.STAT_GPU_COUNT] == 1
    [gpu] = sample[telemetry.STAT_GPUS]
    assert gpu[telemetry.GPU_UTIL_PCT] == pytest.approx(20.4)
    assert gpu[telemetry.GPU_MEM_USED_B] is None


def test_a_jetson_whose_nvml_sample_comes_back_empty_still_reads_one_sysfs_gpu(tmp_path, monkeypatch):
    _install_fake_nvml(monkeypatch)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetCount', lambda: 3)
    load = tmp_path / 'load'
    load.write_text('500\n')
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}', tegra_gpu_load_path=load)
    sample = sampler._sample()
    sampler._nvml.shutdown()
    assert sample[telemetry.STAT_GPU_COUNT] == 1
    assert [gpu[telemetry.GPU_UTIL_PCT] for gpu in sample[telemetry.STAT_GPUS]] == [pytest.approx(50.0)]


def test_a_box_with_no_sysfs_gpu_load_reads_its_gpus_through_nvml(tmp_path, monkeypatch):
    _install_fake_nvml(monkeypatch)
    sampler = telemetry.StatsSampler(
        tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}', tegra_gpu_load_path=tmp_path / 'absent'
    )
    sample = sampler._sample()
    sampler._nvml.shutdown()
    assert [gpu[telemetry.GPU_UTIL_PCT] for gpu in sample[telemetry.STAT_GPUS]] == [_FakeUtil.gpu]


def test_a_box_with_no_sysfs_gpu_load_keeps_an_empty_gpu_list(tmp_path, monkeypatch):
    _refuse_nvml_utilisation(monkeypatch)
    sampler = telemetry.StatsSampler(
        tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}', tegra_gpu_load_path=tmp_path / 'absent'
    )
    sample = sampler._sample()
    sampler._nvml.shutdown()
    assert sample[telemetry.STAT_GPUS] == []


class _UnreadableProc:
    """A child whose every read raises ``error``."""

    pid = 4_000_001

    def __init__(self, error: type[psutil.Error]):
        self._error = error

    def cpu_percent(self, interval=None):
        raise self._error(self.pid)

    def memory_info(self):
        raise self._error(self.pid)


def test_stats_sample_drops_a_child_that_exits_before_its_first_read(tmp_path, monkeypatch, caplog):
    """A child can exit between the tree listing and its first CPU read. The sample keeps the rest of the tree."""
    _install_fake_nvml(monkeypatch)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    monkeypatch.setattr(sampler._self, 'children', lambda recursive=False: [_UnreadableProc(psutil.NoSuchProcess)])

    with caplog.at_level(logging.ERROR, logger=telemetry.__name__):
        sample = sampler._sample()

    assert sample[telemetry.STAT_RSS_PROC_B] > 0
    assert _UnreadableProc.pid not in sampler._proc_cache
    assert not caplog.records
    sampler._nvml.shutdown()


def test_stats_sample_logs_a_child_it_may_not_read(tmp_path, monkeypatch, caplog):
    """A child that refuses a read is left out of the sample, and the log names it."""
    _install_fake_nvml(monkeypatch)
    sampler = telemetry.StatsSampler(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')
    monkeypatch.setattr(sampler._self, 'children', lambda recursive=False: [_UnreadableProc(psutil.AccessDenied)])

    with caplog.at_level(logging.ERROR, logger=telemetry.__name__):
        sample = sampler._sample()

    assert sample[telemetry.STAT_RSS_PROC_B] > 0
    assert _UnreadableProc.pid not in sampler._proc_cache
    assert str(_UnreadableProc.pid) in caplog.records[0].getMessage()
    sampler._nvml.shutdown()


@contextmanager
def _sleeping_child():
    # The child runs its code only after its start-up exec, so its OS name is final once it prints.
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; print(flush=True); time.sleep(60)'], stdout=subprocess.PIPE
    )
    try:
        assert child.stdout is not None
        child.stdout.readline()
        yield child
    finally:
        child.kill()
        child.wait()


def _first_written_sample(path, process_names=None):
    """Run a sampler until it writes one sample to ``path``, and return that sample as the reader parses it."""
    deadline = time.monotonic() + 5.0
    with telemetry.StatsSampler(path, hz=200.0, process_names=process_names):
        while not path.exists() or not path.read_text():
            assert time.monotonic() < deadline, 'the sampler wrote no sample within 5 s'
            time.sleep(0.01)
    return next(telemetry.read_stats(path))


def test_stats_sample_lists_each_process_under_the_callers_name(tmp_path, monkeypatch):
    _install_fake_nvml(monkeypatch)
    with _sleeping_child() as child:
        sample = _first_written_sample(
            tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}', process_names=lambda: {child.pid: 'Camera.run'}
        )

    procs = {proc[telemetry.PROC_PID]: proc for proc in sample[telemetry.STAT_PROCS]}
    assert procs[child.pid][telemetry.PROC_NAME] == 'Camera.run'
    assert procs[child.pid][telemetry.PROC_RSS_B] > 0
    assert procs[os.getpid()][telemetry.PROC_NAME] == psutil.Process().name()
    assert sample[telemetry.STAT_RSS_PROC_B] == sum(proc[telemetry.PROC_RSS_B] for proc in procs.values())
    assert sample[telemetry.STAT_CPU_PROC_PCT] == sum(proc[telemetry.PROC_CPU_PCT] for proc in procs.values())


def test_stats_sample_names_a_process_by_the_os_without_a_mapping(tmp_path, monkeypatch):
    _install_fake_nvml(monkeypatch)
    with _sleeping_child() as child:
        os_name = psutil.Process(child.pid).name()
        sample = _first_written_sample(tmp_path / f'{HARNESS_PROCESS}{telemetry.STATS_SUFFIX}')

    procs = {proc[telemetry.PROC_PID]: proc for proc in sample[telemetry.STAT_PROCS]}
    assert procs[child.pid][telemetry.PROC_NAME] == os_name


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


def test_bind_from_env_is_inert_without_the_env_vars(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_TELEMETRY_DIR, raising=False)
    monkeypatch.delenv(ENV_RUN_ID, raising=False)
    with telemetry.bind_from_env(HARNESS_PROCESS):
        with telemetry.span('client'):
            pass
    assert not (tmp_path / telemetry.TELEMETRY_SUBDIR).exists()


def _run_id(path):
    return _resource_attrs(path)[telemetry.ATTR_RUN_ID]


def _env_sidecars(tmp_path):
    """The sidecars ``bind_from_env`` wrote under a telemetry dir."""
    return sorted((tmp_path / telemetry.TELEMETRY_SUBDIR).glob(f'*{telemetry.SPANS_SUFFIX}'))


def test_bind_from_env_mints_a_run_id_when_none_is_given(tmp_path, monkeypatch):
    """A run id left unset is minted, so every record stamps one for the reduce to group by."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.delenv(ENV_RUN_ID, raising=False)
    with telemetry.bind_from_env(HARNESS_PROCESS):
        with telemetry.span('client'):
            pass
    (path,) = _env_sidecars(tmp_path)
    assert _run_id(path)


def test_bind_from_env_records_under_the_process_it_names(tmp_path, monkeypatch):
    """An attended rollout sets two env vars and gets the sidecar the eval CLI writes."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.setenv(ENV_RUN_ID, 'rollout-1')
    with telemetry.bind_from_env(HARNESS_PROCESS):
        with telemetry.span('client'):
            pass
    (path,) = _env_sidecars(tmp_path)
    assert _spans_by_name(path)['client'].process == HARNESS_PROCESS
    assert _run_id(path) == 'rollout-1'


# Module scope: a World pickles a background loop into the child that runs it.
def _one_span_then_idle(stop_reader, clock):
    with telemetry.span('driver.tick'):
        pass
    telemetry.force_flush()
    while not stop_reader.read().data:
        yield pimm.Sleep(0.01)


def test_a_background_process_writes_its_own_sidecar_under_the_run_id(tmp_path, monkeypatch):
    """A World that binds each child from the environment gives a driver its own file, apart from the harness's."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.setenv(ENV_RUN_ID, 'rollout-1')
    path = tmp_path / telemetry.TELEMETRY_SUBDIR / f'_one_span_then_idle{telemetry.SPANS_SUFFIX}'
    with pimm.World(process_scope=telemetry.bind_from_env) as world:
        world.start_in_subprocess(_one_span_then_idle)
        deadline = time.monotonic() + 30
        while not path.exists() or not path.read_text():
            assert time.monotonic() < deadline, 'the background process wrote no sidecar'
            time.sleep(0.01)
    assert _spans_by_name(path)['driver.tick'].process == '_one_span_then_idle'
    assert _run_id(path) == 'rollout-1'
    assert _env_sidecars(tmp_path) == [path]


def test_one_run_id_gives_a_background_process_the_id_of_the_process_that_spawned_it(tmp_path, monkeypatch):
    """With no run id set, a child binding from the environment would mint its own and join no run."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.delenv(ENV_RUN_ID, raising=False)
    path = tmp_path / telemetry.TELEMETRY_SUBDIR / f'_one_span_then_idle{telemetry.SPANS_SUFFIX}'
    with telemetry.one_run_id(), pimm.World(process_scope=telemetry.bind_from_env) as world:
        run_id = os.environ[ENV_RUN_ID]
        world.start_in_subprocess(_one_span_then_idle)
        deadline = time.monotonic() + 30
        while not path.exists() or not path.read_text():
            assert time.monotonic() < deadline, 'the background process wrote no sidecar'
            time.sleep(0.01)
    assert _run_id(path) == run_id
    assert ENV_RUN_ID not in os.environ


def test_one_run_id_keeps_a_run_id_set_before_it_and_restores_it(monkeypatch):
    monkeypatch.setenv(ENV_RUN_ID, 'given')
    with telemetry.one_run_id():
        assert os.environ[ENV_RUN_ID] == 'given'
        os.environ[ENV_RUN_ID] = 'changed in the block'
    assert os.environ[ENV_RUN_ID] == 'given'


def test_one_run_id_replaces_an_empty_run_id_and_restores_it(monkeypatch):
    """bind_from_env reads an empty run id as absent, so one_run_id must hold a real id for it."""
    monkeypatch.setenv(ENV_RUN_ID, '')
    with telemetry.one_run_id():
        assert os.environ[ENV_RUN_ID]
    assert os.environ[ENV_RUN_ID] == ''


def test_bind_from_env_records_nothing_when_the_sidecar_cannot_open(tmp_path, monkeypatch, caplog):
    """Every recording run binds from the environment, so a sidecar that refuses to open must cost its
    timings and not the run. The warning names the path, so the loss is in the log rather than silent."""
    blocked = tmp_path / 'blocked'
    blocked.write_text('a file where the telemetry directory would go')
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(blocked))
    monkeypatch.delenv(ENV_RUN_ID, raising=False)

    with caplog.at_level(logging.WARNING, logger='positronic.telemetry'):
        with telemetry.bind_from_env(HARNESS_PROCESS):
            with telemetry.span('client'):
                pass

    assert str(blocked) in caplog.text
    assert blocked.read_text() == 'a file where the telemetry directory would go'


def test_bind_from_env_without_the_telemetry_extra_runs_on(tmp_path, monkeypatch, caplog, without_telemetry_extra):
    """An install with no SDK records no spans and still runs its episodes — the extra rides in the dev
    group, so an image built lean is the case this meets."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.delenv(ENV_RUN_ID, raising=False)

    with caplog.at_level(logging.WARNING, logger='positronic.telemetry'):
        with telemetry.bind_from_env(HARNESS_PROCESS):
            with telemetry.span('client'):
                pass

    assert 'telemetry extra' in caplog.text
    assert '--timing' not in caplog.text, 'the run did not ask for timing'
    assert 'Traceback' not in caplog.text
    assert _env_sidecars(tmp_path) == []


def test_bind_from_env_defers_to_a_provider_already_bound(tmp_path, monkeypatch):
    """The CLI owns the lifecycle under ``eval run --timing``, and keeps it."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / 'ignored' / telemetry.TELEMETRY_SUBDIR))
    monkeypatch.setenv(ENV_RUN_ID, 'from-env')
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'from-the-cli'):
        with telemetry.bind_from_env(HARNESS_PROCESS):
            with telemetry.span('client'):
                pass
        with telemetry.span('after'):
            pass
    path = telemetry.spans_path(tmp_path, HARNESS_PROCESS)
    assert set(_spans_by_name(path)) == {'client', 'after'}
    assert _run_id(path) == 'from-the-cli'
    assert not (tmp_path / 'ignored').exists()


def _record_one_of_each(level):
    """Open a call span holding a step span, and record a step span by its bounds, through each helper."""
    with _anchored('episode'):
        with telemetry.span('call'):
            with telemetry.span('step', level):
                pass
        telemetry.record_span('recorded', time.time_ns(), time.time_ns(), level)
        telemetry.traced('traced', level)(lambda: None)()
        telemetry.start_span('started', level).end()


@pytest.mark.parametrize(
    ('bound', 'names'),
    [
        (telemetry.Level.CALL, ['call', 'episode']),
        (telemetry.Level.STEP, ['call', 'episode', 'recorded', 'started', 'step', 'traced']),
    ],
)
def test_a_bound_process_records_the_spans_up_to_its_level(tmp_path, bound, names):
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'levels', level=bound):
        _record_one_of_each(telemetry.Level.STEP)
    spans = telemetry.read_spans(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert sorted(span.name for span in spans) == names


def test_a_span_above_the_level_builds_no_attributes(tmp_path):
    """An object that json cannot encode raises when a span encodes its attributes, so a quiet run proves that
    nothing encoded them."""
    unencodable = object()
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'inert', level=telemetry.Level.CALL):
        with telemetry.span('step', telemetry.Level.STEP, value=unencodable) as span:
            telemetry.set_attrs(span, value=unencodable)
        telemetry.record_span('step', 0, 1, telemetry.Level.STEP, value=unencodable)
        telemetry.start_span('step', telemetry.Level.STEP, value=unencodable).end()
        first = telemetry.span('step', telemetry.Level.STEP)
        assert telemetry.span('other', telemetry.Level.STEP) is first
        assert not telemetry.enabled(telemetry.Level.STEP)
        assert telemetry.enabled(telemetry.Level.CALL)


def test_a_call_span_inside_an_unrecorded_step_span_parents_to_the_anchor(tmp_path):
    with telemetry.bind(tmp_path, HARNESS_PROCESS, 'parent', level=telemetry.Level.CALL):
        with _anchored('episode'):
            with telemetry.span('step', telemetry.Level.STEP):
                with telemetry.span('call'):
                    pass
    spans = _spans_by_name(telemetry.spans_path(tmp_path, HARNESS_PROCESS))
    assert spans['call'].parent_id == spans['episode'].span_id


def test_a_timing_sink_outside_a_bind_gets_the_step_spans():
    timings = []
    with telemetry.timings_to(lambda *args: timings.append(args)):
        with telemetry.span('step', telemetry.Level.STEP):
            pass
    assert [name for name, _, _ in timings] == ['step']


def test_bind_records_every_span_unless_the_environment_names_a_level(tmp_path, monkeypatch):
    monkeypatch.delenv(telemetry.ENV_TELEMETRY_LEVEL, raising=False)
    with telemetry.bind(tmp_path / 'default', HARNESS_PROCESS, 'default'):
        assert telemetry.enabled(telemetry.Level.STEP)
    monkeypatch.setenv(telemetry.ENV_TELEMETRY_LEVEL, 'call')
    with telemetry.bind(tmp_path / 'named', HARNESS_PROCESS, 'named'):
        assert not telemetry.enabled(telemetry.Level.STEP)


@pytest.mark.parametrize(('name', 'step_spans'), [(None, False), ('call', False), ('step', True)])
def test_bind_from_env_records_the_step_spans_only_when_the_environment_asks(tmp_path, monkeypatch, name, step_spans):
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path))
    if name is None:
        monkeypatch.delenv(telemetry.ENV_TELEMETRY_LEVEL, raising=False)
    else:
        monkeypatch.setenv(telemetry.ENV_TELEMETRY_LEVEL, name)
    with telemetry.bind_from_env(HARNESS_PROCESS):
        assert telemetry.enabled(telemetry.Level.CALL)
        assert telemetry.enabled(telemetry.Level.STEP) is step_spans
    assert telemetry.enabled(telemetry.Level.STEP) is False


def test_bind_from_env_refuses_a_level_name_that_is_not_valid(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path))
    monkeypatch.setenv(telemetry.ENV_TELEMETRY_LEVEL, 'full')
    with pytest.raises(ValueError, match='call, step'), telemetry.bind_from_env(HARNESS_PROCESS):
        pass
