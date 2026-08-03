import json
import os
import sys
import time

import pynvml
import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

from positronic import telemetry


def _spans_by_name(path):
    return {rec.name: rec for rec in telemetry.read_spans(path)}


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
    with telemetry.span(telemetry.SPAN_RESET):
        pass
    episode = telemetry.begin_episode()
    telemetry.end_episode(episode)


def test_recording_without_the_telemetry_extra_names_the_extra(tmp_path, without_telemetry_extra):
    """Both entry points ``--timing`` reaches fail with the install command, not an ImportError on a package
    name the operator has no reason to connect to ``--timing``."""
    with pytest.raises(RuntimeError, match='telemetry extra'):
        with telemetry.bind(tmp_path, 'harness', 'run-no-extra'):
            pass
    with pytest.raises(RuntimeError, match='telemetry extra'):
        telemetry.StatsSampler(tmp_path / 'harness.stats.jsonl')


def test_nested_spans_round_trip(tmp_path):
    """A bound run's nested spans parse back with the taxonomy's parenting (materialize under env.step, both
    under episode, episode under the pass), hex ids, and decoded attributes."""
    with telemetry.bind(tmp_path, 'harness', 'run-xyz'):
        with telemetry.eval_pass(**{'run.id': 'run-xyz', 'policy': 'stub'}):
            episode = telemetry.begin_episode(**{telemetry.ATTR_EPISODE_INDEX: 0})
            with telemetry.span(telemetry.SPAN_RESET):
                pass
            with telemetry.span(telemetry.SPAN_ENV_STEP):
                with telemetry.span(telemetry.SPAN_MATERIALIZE):
                    pass
            telemetry.end_episode(episode, **{telemetry.ATTR_EPISODE_STEPS: 5, telemetry.ATTR_EPISODE_VIRTUAL_S: 1.5})

    spans_path = tmp_path / 'telemetry' / 'harness.spans.jsonl'
    spans = _spans_by_name(spans_path)
    assert set(spans) == {
        telemetry.SPAN_EVAL_PASS,
        telemetry.SPAN_EPISODE,
        telemetry.SPAN_RESET,
        telemetry.SPAN_ENV_STEP,
        telemetry.SPAN_MATERIALIZE,
    }

    # Hex ids: 16 nibbles for a span id, and every id parses as hex.
    for rec in spans.values():
        assert len(rec.span_id) == 16
        int(rec.span_id, 16)

    assert spans[telemetry.SPAN_EVAL_PASS].parent_id is None
    assert spans[telemetry.SPAN_EPISODE].parent_id == spans[telemetry.SPAN_EVAL_PASS].span_id
    assert spans[telemetry.SPAN_RESET].parent_id == spans[telemetry.SPAN_EPISODE].span_id
    assert spans[telemetry.SPAN_ENV_STEP].parent_id == spans[telemetry.SPAN_EPISODE].span_id
    assert spans[telemetry.SPAN_MATERIALIZE].parent_id == spans[telemetry.SPAN_ENV_STEP].span_id

    assert spans[telemetry.SPAN_EVAL_PASS].attrs['policy'] == 'stub'
    assert spans[telemetry.SPAN_EPISODE].attrs[telemetry.ATTR_EPISODE_STEPS] == 5
    assert spans[telemetry.SPAN_EPISODE].attrs[telemetry.ATTR_EPISODE_VIRTUAL_S] == 1.5
    assert spans[telemetry.SPAN_EPISODE].end_ns >= spans[telemetry.SPAN_EPISODE].start_ns


def test_discarded_episode_marked_aborted(tmp_path):
    with telemetry.bind(tmp_path, 'harness', 'run-abort'):
        episode = telemetry.begin_episode(**{telemetry.ATTR_EPISODE_INDEX: 0})
        telemetry.discard_episode(episode)
    spans = _spans_by_name(tmp_path / 'telemetry' / 'harness.spans.jsonl')
    assert spans[telemetry.SPAN_EPISODE].attrs[telemetry.ATTR_EPISODE_ABORTED] is True


def test_unbound_span_is_inert(tmp_path):
    """Off ``--timing`` nothing binds: the span helpers no-op, write no file, and raise nothing."""
    with telemetry.span(telemetry.SPAN_RESET):
        pass
    episode = telemetry.begin_episode()
    telemetry.end_episode(episode)
    assert not (tmp_path / 'telemetry').exists()


def test_resource_carries_process_identity(tmp_path):
    """Every span document's resource block names the run and the writing process, so a sidecar identifies
    itself without a second file."""
    with telemetry.bind(tmp_path, 'env', 'run-1'):
        with telemetry.span(telemetry.SPAN_RESET):
            pass

    line = json.loads((tmp_path / 'telemetry' / 'env.spans.jsonl').read_text().splitlines()[0])
    attrs = telemetry._decode_attrs(line['resourceSpans'][0]['resource']['attributes'])
    assert attrs['run.id'] == 'run-1'
    assert attrs['process.name'] == 'env'
    assert attrs['process.pid'] == os.getpid()


class _FakeUtil:
    gpu = 42


class _FakeMem:
    used = 3 * 1024**3
    total = 8 * 1024**3


class _FakeProc:
    def __init__(self, pid, used):
        self.pid = pid
        self.usedGpuMemory = used


def test_failed_pass_exported_and_stamped(tmp_path):
    """A sweep that dies mid-pass still exports its pass span — the partial window is real data — stamped
    ``pass.failed`` so the reduce can name the mix instead of silently folding it in."""
    with telemetry.bind(tmp_path, 'harness', 'run-fail'):
        with pytest.raises(RuntimeError):
            with telemetry.eval_pass(**{'run.id': 'run-fail'}):
                raise RuntimeError('sim died')

    spans = {s.name: s for s in telemetry.read_spans(tmp_path / 'telemetry' / 'harness.spans.jsonl')}
    assert spans[telemetry.SPAN_EVAL_PASS].attrs.get(telemetry.ATTR_PASS_FAILED) is True


def test_unbound_span_never_touches_global_tracer(monkeypatch):
    """Unbound telemetry is a PRIVATE no-op: a host application may have configured OTel's global tracer
    provider, and an untimed run must not export spans (or trial attributes) through it."""

    def _boom(*args, **kwargs):
        raise AssertionError('unbound telemetry must not consult the global tracer provider')

    monkeypatch.setattr(telemetry.trace, 'get_tracer', _boom)
    with telemetry.span(telemetry.SPAN_RESET, **{telemetry.ATTR_EPISODE_INDEX: 0}):
        pass


def test_per_tick_span_ignores_foreign_ambient_span(tmp_path):
    """A host application's current OTel span (a different provider, a different trace) must not adopt our
    per-tick spans: they parent to the episode in flight, so the report still sees them."""
    foreign = SdkTracerProvider()
    with telemetry.bind(tmp_path, 'harness', 'run-foreign'):
        episode = telemetry.begin_episode(**{telemetry.ATTR_EPISODE_INDEX: 0})
        with foreign.get_tracer('host-app').start_as_current_span('host-span'):
            with telemetry.span(telemetry.SPAN_ENV_STEP):
                pass
        telemetry.end_episode(episode, **{telemetry.ATTR_EPISODE_STEPS: 1, telemetry.ATTR_EPISODE_VIRTUAL_S: 0.0})

    spans = {s.name: s for s in telemetry.read_spans(tmp_path / 'telemetry' / 'harness.spans.jsonl')}
    assert 'host-span' not in spans  # the foreign span belongs to the host's provider, not our file
    assert spans[telemetry.SPAN_ENV_STEP].parent_id == spans[telemetry.SPAN_EPISODE].span_id


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
    sampler = telemetry.StatsSampler(tmp_path / 'harness.stats.jsonl')
    sample = sampler._sample()
    assert set(sample) >= {'t_ns', 'cpu_sys_pct', 'iowait_pct', 'mem_sys_used_b', 'cpu_proc_pct', 'rss_proc_b', 'gpus'}
    assert len(sample['gpus']) == 1
    gpu = sample['gpus'][0]
    assert gpu['i'] == 0
    assert gpu['util_pct'] == 42.0
    assert gpu['mem_used_b'] == 3 * 1024**3
    assert gpu['power_w'] == 150.0
    # The same pid in the compute and graphics lists merges by max — drivers commonly report the process
    # total in each, so summing both entries would double-count.
    assert gpu['proc_mem_b'] == 1234 * 1024**2
    assert gpu['proc_util_pct'] is None
    sampler._nvml.shutdown()


def test_stats_sample_treats_nvml_sentinel_as_unavailable(tmp_path, monkeypatch):
    """A driver that cannot attribute a process's GPU memory returns NVML's uint64 sentinel (~18 EiB), not
    ``None`` — recording it verbatim would publish garbage as the eval's peak process VRAM. The device reads
    unavailable (``proc_mem_b`` None) instead (Codex on PR #531)."""
    _install_fake_nvml(monkeypatch)
    sentinel_proc = _FakeProc(os.getpid(), telemetry._NVML_VALUE_NOT_AVAILABLE)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', lambda h: [sentinel_proc])
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses', lambda h: [])
    sampler = telemetry.StatsSampler(tmp_path / 'harness.stats.jsonl')
    sample = sampler._sample()
    gpu = sample['gpus'][0]
    assert gpu['proc_mem_b'] is None  # pre-fix, the sentinel was int()-cast and recorded as ~1.8e19 bytes
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
    sampler = telemetry.StatsSampler(tmp_path / 'harness.stats.jsonl')
    sample = sampler._sample()
    assert [gpu['i'] for gpu in sample['gpus']] == [1]
    assert isinstance(sample['cpu_sys_pct'], float)
    sampler._nvml.shutdown()


def test_stats_sample_without_gpu(tmp_path, monkeypatch):
    def _raise():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(pynvml, 'nvmlInit', _raise)
    sampler = telemetry.StatsSampler(tmp_path / 'harness.stats.jsonl')
    sample = sampler._sample()
    assert sample['gpus'] == []
    assert isinstance(sample['cpu_sys_pct'], float)
    assert isinstance(sample['rss_proc_b'], int)


def test_stats_sampler_thread_writes_lines(tmp_path, monkeypatch):
    def _raise():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(pynvml, 'nvmlInit', _raise)
    path = tmp_path / 'harness.stats.jsonl'
    with telemetry.StatsSampler(path, hz=200.0):
        time.sleep(0.1)
    samples = list(telemetry.read_stats(path))
    assert samples, 'the sampler thread wrote at least one line'
    assert all('t_ns' in sample for sample in samples)


def test_bind_seals_truncated_predecessor_line(tmp_path):
    """A killed run can leave ``<process>.spans.jsonl`` ending in a truncated fragment with no trailing newline.
    The next bind into the same directory seals that line before its exporter appends, so the first new record
    starts a fresh line instead of concatenating onto the fragment and being skipped along with it (Codex on
    PR #531)."""
    spans_path = tmp_path / 'telemetry' / 'harness.spans.jsonl'

    with telemetry.bind(tmp_path, 'harness', 'run-1'):
        with telemetry.span(telemetry.SPAN_RESET):
            pass
    with open(spans_path, 'a') as file:
        file.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"nam')  # crash mid-write, no trailing newline

    with telemetry.bind(tmp_path, 'harness', 'run-2'):
        with telemetry.span(telemetry.SPAN_MATERIALIZE):
            pass

    names = [rec.name for rec in telemetry.read_spans(spans_path)]
    assert telemetry.SPAN_MATERIALIZE in names  # pre-fix, the new record merges into the fragment and is skipped
    assert telemetry.SPAN_RESET in names  # the pre-existing valid span still reads back


def test_readers_tolerate_truncated_final_line(tmp_path):
    spans_path = tmp_path / 'telemetry' / 'harness.spans.jsonl'
    with telemetry.bind(tmp_path, 'harness', 'run-trunc'):
        with telemetry.span(telemetry.SPAN_RESET):
            pass
    with open(spans_path, 'a') as file:
        file.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"nam')  # crash mid-write
    names = [rec.name for rec in telemetry.read_spans(spans_path)]
    assert names == [telemetry.SPAN_RESET]

    stats_path = tmp_path / 'stats.jsonl'
    stats_path.write_text('{"t_ns": 1, "gpus": []}\n{"t_ns": 2, "cpu_sy')
    stats = list(telemetry.read_stats(stats_path))
    assert [sample['t_ns'] for sample in stats] == [1]
