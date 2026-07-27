import json
import os
import time

import pynvml
import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

from positronic import telemetry


def _spans_by_name(path):
    return {rec.name: rec for rec in telemetry.read_spans(path)}


def test_nested_spans_round_trip(tmp_path):
    """A bound run's nested spans parse back with the taxonomy's parenting (materialize under env.step, both
    under episode, episode under the pass), hex ids, and decoded attributes."""
    with telemetry.bind(tmp_path, 'harness', 'run-xyz'):
        with telemetry.eval_pass(**{'run.id': 'run-xyz', 'policy': 'stub'}):
            episode = telemetry.begin_episode(**{'episode.index': 0})
            with telemetry.span('reset'):
                pass
            with telemetry.span('env.step'):
                with telemetry.span('materialize'):
                    pass
            telemetry.end_episode(episode, **{'episode.steps': 5, 'episode.virtual_s': 1.5})

    spans_path = tmp_path / 'telemetry' / 'harness.spans.jsonl'
    spans = _spans_by_name(spans_path)
    assert set(spans) == {'eval.pass', 'episode', 'reset', 'env.step', 'materialize'}

    # Hex ids: 16 nibbles for a span id, and every id parses as hex.
    for rec in spans.values():
        assert len(rec.span_id) == 16
        int(rec.span_id, 16)

    assert spans['eval.pass'].parent_id is None
    assert spans['episode'].parent_id == spans['eval.pass'].span_id
    assert spans['reset'].parent_id == spans['episode'].span_id
    assert spans['env.step'].parent_id == spans['episode'].span_id
    assert spans['materialize'].parent_id == spans['env.step'].span_id

    assert spans['eval.pass'].attrs['policy'] == 'stub'
    assert spans['episode'].attrs['episode.steps'] == 5
    assert spans['episode'].attrs['episode.virtual_s'] == 1.5
    assert spans['episode'].end_ns >= spans['episode'].start_ns


def test_discarded_episode_marked_aborted(tmp_path):
    with telemetry.bind(tmp_path, 'harness', 'run-abort'):
        episode = telemetry.begin_episode(**{'episode.index': 0})
        telemetry.discard_episode(episode)
    spans = _spans_by_name(tmp_path / 'telemetry' / 'harness.spans.jsonl')
    assert spans['episode'].attrs['episode.aborted'] is True


def test_unbound_span_is_inert(tmp_path):
    """Off ``--timing`` nothing binds: the span helpers no-op, write no file, and raise nothing."""
    with telemetry.span('reset'):
        pass
    episode = telemetry.begin_episode()
    telemetry.end_episode(episode)
    assert not (tmp_path / 'telemetry').exists()


def test_meta_written(tmp_path):
    with telemetry.bind(tmp_path, 'env', 'run-1'):
        pass
    meta = json.loads((tmp_path / 'telemetry' / 'env.meta.json').read_text())
    assert meta['schema'] == 1
    assert meta['run_id'] == 'run-1'
    assert meta['process'] == 'env'
    assert meta['pid'] == os.getpid()


class _FakeUtil:
    gpu = 42


class _FakeMem:
    used = 3 * 1024**3
    total = 8 * 1024**3


class _FakeProc:
    def __init__(self, pid, used):
        self.pid = pid
        self.usedGpuMemory = used


def test_append_after_truncated_line_keeps_new_records(tmp_path):
    """Reusing a directory whose previous run died mid-write must not merge the new run's first record into
    the truncated fragment — the writer seals the fragment with a newline before appending."""
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    path = telemetry_dir / 'harness.spans.jsonl'
    path.write_text('{"resourceSpans": [{"scopeSp')  # a killed run's partial write, no trailing newline

    with telemetry.bind(tmp_path, 'harness', 'run-resume'):
        with telemetry.span('reset'):
            pass

    assert [s.name for s in telemetry.read_spans(path)] == ['reset']


def test_failed_pass_exported_and_stamped(tmp_path):
    """A sweep that dies mid-pass still exports its pass span — the partial window is real data — stamped
    ``pass.failed`` so the reduce can name the mix instead of silently folding it in."""
    with telemetry.bind(tmp_path, 'harness', 'run-fail'):
        with pytest.raises(RuntimeError):
            with telemetry.eval_pass(**{'run.id': 'run-fail'}):
                raise RuntimeError('sim died')

    spans = {s.name: s for s in telemetry.read_spans(tmp_path / 'telemetry' / 'harness.spans.jsonl')}
    assert spans['eval.pass'].attrs.get('pass.failed') is True


def test_unbound_span_never_touches_global_tracer(monkeypatch):
    """Unbound telemetry is a PRIVATE no-op: a host application may have configured OTel's global tracer
    provider, and an untimed run must not export spans (or trial attributes) through it."""

    def _boom(*args, **kwargs):
        raise AssertionError('unbound telemetry must not consult the global tracer provider')

    monkeypatch.setattr(telemetry.trace, 'get_tracer', _boom)
    with telemetry.span('reset', **{'episode.index': 0}):
        pass


def test_per_tick_span_ignores_foreign_ambient_span(tmp_path):
    """A host application's current OTel span (a different provider, a different trace) must not adopt our
    per-tick spans: they parent to the episode in flight, so the report still sees them."""
    foreign = SdkTracerProvider()
    with telemetry.bind(tmp_path, 'harness', 'run-foreign'):
        episode = telemetry.begin_episode(**{'episode.index': 0})
        with foreign.get_tracer('host-app').start_as_current_span('host-span'):
            with telemetry.span('env.step'):
                pass
        telemetry.end_episode(episode, **{'episode.steps': 1, 'episode.virtual_s': 0.0})

    spans = {s.name: s for s in telemetry.read_spans(tmp_path / 'telemetry' / 'harness.spans.jsonl')}
    assert 'host-span' not in spans  # the foreign span belongs to the host's provider, not our file
    assert spans['env.step'].parent_id == spans['episode'].span_id


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
    assert gpu['proc_mem_b'] == (1234 + 100) * 1024**2  # compute + graphics contexts of this process tree
    assert gpu['proc_util_pct'] is None
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


def test_readers_tolerate_truncated_final_line(tmp_path):
    spans_path = tmp_path / 'telemetry' / 'harness.spans.jsonl'
    with telemetry.bind(tmp_path, 'harness', 'run-trunc'):
        with telemetry.span('reset'):
            pass
    with open(spans_path, 'a') as file:
        file.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"nam')  # crash mid-write
    names = [rec.name for rec in telemetry.read_spans(spans_path)]
    assert names == ['reset']

    stats_path = tmp_path / 'stats.jsonl'
    stats_path.write_text('{"t_ns": 1, "gpus": []}\n{"t_ns": 2, "cpu_sy')
    stats = list(telemetry.read_stats(stats_path))
    assert [sample['t_ns'] for sample in stats] == [1]
