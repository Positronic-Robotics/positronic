from pathlib import Path

import pytest

from positronic import eval_timing


def test_record_env_phases_maps_each_key_to_a_timing_env_signal(tmp_path: Path):
    """``record_env_phases`` namespaces each env-reported phase into a ``timing.env_<phase>`` signal and sums
    repeated reports within a drain, with no phase set baked into the telemetry module (comment 2)."""
    with eval_timing.bind(tmp_path):
        eval_timing.begin_episode()
        eval_timing.record_env_phases({'physics_s': 0.5, 'render_s': 0.25})
        eval_timing.record_env_phases({'physics_s': 0.125, 'server_other_s': 0.4})
        pairs = dict(eval_timing.drain_signal_items())
    assert pairs == pytest.approx({
        'timing.env_physics_s': 0.625,
        'timing.env_render_s': 0.25,
        'timing.env_server_other_s': 0.4,
    })
