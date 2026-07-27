import pytest

from positronic import eval_timing
from positronic.drivers.gpu_monitor import GpuMonitor, _GpuSample


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
