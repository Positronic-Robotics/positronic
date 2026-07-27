import json
import os
import time

import pynvml

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
    assert gpu['proc_mem_b'] == 1234 * 1024**2  # this test's own pid is in the process tree
    assert gpu['proc_util_pct'] is None
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
