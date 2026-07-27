from datetime import UTC, datetime

import pytest

from positronic import eval_timing
from positronic.cli.eval.timing_report import _load_episodes
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import gpu_monitor
from positronic.drivers.gpu_monitor import GpuMonitor, _GpuSample
from positronic.eval_timing import FINISHED_AT_KEY, RESET_S_KEY, WALL_S_KEY


def test_record_env_phases_maps_each_key_to_a_timing_env_signal():
    """``record_env_phases`` namespaces each env-reported phase into a ``timing.env_<phase>`` signal and sums
    repeated reports within a drain, with no phase set baked into the telemetry module (comment 2)."""
    with eval_timing.bind():
        eval_timing.begin_episode()
        eval_timing.record_env_phases({'physics_s': 0.5, 'render_s': 0.25})
        eval_timing.record_env_phases({'physics_s': 0.125, 'server_other_s': 0.4})
        pairs = dict(eval_timing.drain_signal_items())
    assert pairs == pytest.approx({
        'timing.env_physics_s': 0.625,
        'timing.env_render_s': 0.25,
        'timing.env_server_other_s': 0.4,
    })


def test_gpu_monitor_retains_all_buffered_probes():
    """A synchronous span longer than the sampling interval buffers several probes before the cooperative loop
    runs; all are retained as one batch (not collapsed to the latest — Codex P1), each keeping its real capture
    wall-ns, at strictly increasing timestamps placed at the drain instant."""
    monitor = GpuMonitor()
    monitor._buffer = [
        _GpuSample(util_pct=10.0, mem_mib=100.0, proc_mem_mib=50.0, wall_ns=1),
        _GpuSample(util_pct=20.0, mem_mib=200.0, proc_mem_mib=60.0, wall_ns=2),
        _GpuSample(util_pct=30.0, mem_mib=300.0, proc_mem_mib=70.0, wall_ns=3),
    ]
    batch = monitor._drain_batch(now_ns=1000)
    assert [sample.value['_util'] for sample in batch] == [10.0, 20.0, 30.0]
    assert [sample.value['_wall_ns'] for sample in batch] == [1, 2, 3]
    assert [sample.ts for sample in batch] == [1000, 1001, 1002]
    assert monitor._buffer == []
    assert monitor._drain_batch(now_ns=1000) == []  # an empty buffer on the next tick emits nothing


def test_gpu_sample_none_on_nonnumeric_metric(monkeypatch):
    """A metric reported as `[Not Supported]` / `N/A` (some MIG configs) is a successful nvidia-smi call with
    non-numeric output; it must yield None (skip the sample), not raise in the daemon thread and permanently
    stop sampling (Codex P2)."""
    monkeypatch.setattr(gpu_monitor, '_run_nvidia_smi', lambda args: '[Not Supported], 1024\n')
    assert gpu_monitor._read_gpu_sample('0') is None


def test_timing_report_splits_reused_output_dir_into_runs(tmp_path):
    """Two timed passes appended to one output_dir must not merge into one run — else the wall gap between them
    counts as pass time and understates the real-time factor (Codex P1). Runs split at the invocation window
    each episode was recorded in, read from the run_metadata_<ts>.yaml markers each invocation writes."""

    def invocation_ns(hour):
        return int(datetime(2026, 1, 1, hour, tzinfo=UTC).timestamp()) * 1_000_000_000

    with LocalDatasetWriter(tmp_path) as writer:
        for hour in (0, 2):  # two eval-run invocations, two hours apart, into the same dir
            marker = datetime(2026, 1, 1, hour, tzinfo=UTC).strftime('%Y%m%d_%H%M%S')
            (tmp_path / f'run_metadata_{marker}.yaml').write_text('')
            for episode in range(2):
                created = invocation_ns(hour) + (episode + 1) * 1_000_000_000
                with writer.new_episode(created_ts_ns=created) as ep_writer:
                    ep_writer.append('gpu_load', 0.0, ts_ns=created)
                    ep_writer.set_static(WALL_S_KEY, 1.0)
                    ep_writer.set_static(RESET_S_KEY, 0.1)
                    ep_writer.set_static(FINISHED_AT_KEY, created / 1e9 + 1.0)

    runs, facts = _load_episodes(tmp_path)
    assert [len(run) for run in runs] == [2, 2]  # the two passes stay separate, not merged into one run
    assert len(facts) == 4
