"""Tests for ``eval_timing_shim``: the emitted ``timing.jsonl`` + episode stubs feed ``EpisodeTiming`` and
reduce cleanly through the real ``positronic eval timing-report`` CLI."""

import json
import os
import shutil
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import pytest

from positronic.eval_timing import GPU_LOG_FILENAME, TIMING_FILENAME, EpisodeTiming
from positronic.eval_timing_shim import (
    HOST_LOG_FILENAME,
    SIGNAL_FILENAME,
    TimingShim,
    _cpu_busy_pct,
    _HostSampler,
    _read_meminfo,
    _read_proc_stat,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SECONDS_FIELDS = [f.name for f in fields(EpisodeTiming) if f.name.endswith('_s')]


def _drive_episode(ep, n_steps: int) -> None:
    """Run one synthetic rollout through the shim's phase hooks with small real sleeps for a realistic split."""
    with ep.reset():
        time.sleep(0.002)
    for _ in range(n_steps):
        with ep.policy():
            time.sleep(0.005)
        with ep.env_step():
            time.sleep(0.004)
        ep.add_env_phases(physics_s=0.002, render_s=0.001, server_s=0.0035)
        with ep.record_io():
            time.sleep(0.001)


def _build_synthetic_run(out_dir: Path, *, sample_gpu: bool = False) -> dict:
    """Emit a full run: four scored episodes (two success, one fail, one unscored) plus a discarded abort.

    Returns the facts the assertions need — the expected recorded-episode count and success rate.
    """
    outcomes = [
        {'success': True, 'terminated': True, 'scored': True},
        {'success': True, 'terminated': True, 'scored': True},
        {'success': False, 'terminated': True, 'scored': True},
        {'success': None, 'terminated': None, 'scored': False},  # unscored: excluded from the rate
    ]
    shim = TimingShim(out_dir, task='pick_cube', sample_gpu=sample_gpu)
    with shim.run():
        for trial, outcome in enumerate(outcomes):
            with shim.episode(trial=trial, sim_duration_s=5.0, size_bytes=2 * 1024 * 1024) as ep:
                _drive_episode(ep, n_steps=3)
                ep.set_outcome(**outcome)
        # An abort mid-rollout must not be recorded, in timing.jsonl or as an episode dir.
        with pytest.raises(RuntimeError):
            with shim.episode(trial=99, sim_duration_s=5.0) as ep:
                _drive_episode(ep, n_steps=1)
                raise RuntimeError('injected abort')
    return {'episodes': len(outcomes), 'expected_success_rate': 2 / 3}


def _read_timing(out_dir: Path) -> list[dict]:
    lines = (out_dir / TIMING_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_timing_jsonl_feeds_episode_timing(tmp_path):
    facts = _build_synthetic_run(tmp_path)
    rows = _read_timing(tmp_path)

    # The discarded episode is absent; only the four completed rollouts are recorded.
    assert len(rows) == facts['episodes']

    for row in rows:
        # The line round-trips into the dataclass exactly as the reducer decodes it — no missing/extra keys.
        record = EpisodeTiming(**row)
        assert isinstance(record.task, str)
        assert isinstance(record.trial, int) and not isinstance(record.trial, bool)
        assert isinstance(record.episode_uid, str) and record.episode_uid
        for name in SECONDS_FIELDS:
            value = getattr(record, name)
            assert isinstance(value, float), f'{name} must be float, got {type(value)}'
            assert value >= 0.0, f'{name} negative'
        assert all(isinstance(x, float) for x in record.infer_ms)
        assert len(record.infer_ms) == 3  # one entry per policy call
        # Epoch seconds at seal — a real recent wall clock, not a relative offset.
        assert isinstance(record.finished_at, float) and record.finished_at > 1.7e9

        # Producer invariants the reducer's derived metrics assume.
        assert record.policy_wait_s == pytest.approx(sum(record.infer_ms) / 1000.0, abs=1e-3)
        measured = record.reset_s + record.env_step_s + record.policy_wait_s + record.record_io_s
        assert record.overhead_s == pytest.approx(max(record.wall_s - measured, 0.0), abs=1e-6)
        # env_step server decomposition stays within the client-observed env-step wall.
        assert record.env_server_s <= record.env_step_s + 1e-9
        assert record.env_physics_s + record.env_render_s <= record.env_server_s + 1e-9


def test_dataset_dir_layout_joins_by_uid(tmp_path):
    _build_synthetic_run(tmp_path)
    rows = _read_timing(tmp_path)
    recorded_uids = set()
    for meta_path in sorted(tmp_path.rglob('meta.json')):
        meta = json.loads(meta_path.read_text())
        assert 'uid' in meta and 'duration_ns' in meta
        # 12-digit block/episode names, as load_all_datasets requires.
        assert meta_path.parent.name.isdigit() and len(meta_path.parent.name) == 12
        assert meta_path.parent.parent.name.isdigit() and len(meta_path.parent.parent.name) == 12
        assert (meta_path.parent / 'static.json').exists()
        assert (meta_path.parent / SIGNAL_FILENAME).exists()
        recorded_uids.add(meta['uid'])
    # Every timing record's join key has a matching recorded episode (the reducer fails otherwise).
    assert {row['episode_uid'] for row in rows} == recorded_uids
    assert len(recorded_uids) == len(rows)


def test_finished_episodes_flush_to_disk_before_close(tmp_path):
    # A pass killed mid-run (no close(), no run() wrapper) must still leave every finished rollout's timing
    # on disk: records append as each episode seals rather than buffering until close(). A stale timing.jsonl
    # from a prior pass is truncated lazily on the first flush.
    (tmp_path / TIMING_FILENAME).write_text('stale line from a prior pass\n')
    shim = TimingShim(tmp_path, task='pick_cube', sample_gpu=False, sample_host=False)
    uids = []
    for trial in range(3):
        ep = shim.begin_episode(trial=trial, sim_duration_s=5.0)
        _drive_episode(ep, n_steps=2)
        ep.set_outcome(success=True, terminated=True, scored=True)
        shim.finish_episode(ep)
        uids.append(ep.uid)
        # The record is durable the moment the episode seals — no close() has run yet.
        assert len(_read_timing(tmp_path)) == trial + 1

    rows = _read_timing(tmp_path)
    assert [row['episode_uid'] for row in rows] == uids  # stale line gone, three fresh records in seal order


def test_run_clears_stale_gpu_log_with_sampling_off(tmp_path):
    # The reducer auto-reads any gpu_dmon.log in the dataset dir, so a prior GPU pass's log must not
    # survive into a fresh sample_gpu=False pass.
    (tmp_path / GPU_LOG_FILENAME).write_text('stale dmon samples\n')
    _build_synthetic_run(tmp_path, sample_gpu=False)
    assert not (tmp_path / GPU_LOG_FILENAME).exists()


def test_run_clears_stale_host_log_with_sampling_off(tmp_path):
    # Symmetric with the GPU log: a prior pass's host_stats.log must not survive a sample_host=False pass,
    # else stale CPU/RAM samples sit next to the fresh timing.jsonl and are mistaken for current telemetry.
    (tmp_path / HOST_LOG_FILENAME).write_text('stale host samples\n')
    shim = TimingShim(tmp_path, task='pick_cube', sample_gpu=False, sample_host=False)
    with shim.run():
        pass
    assert not (tmp_path / HOST_LOG_FILENAME).exists()


def test_gpu_sampling_degrades_gracefully(tmp_path):
    _build_synthetic_run(tmp_path, sample_gpu=True)
    gpu_log = tmp_path / GPU_LOG_FILENAME
    if shutil.which('nvidia-smi') is None:
        assert not gpu_log.exists(), 'no GPU log should be written on a box without nvidia-smi'
    else:
        assert gpu_log.exists()


HOST_SAMPLE_KEYS = {'t_mono', 't_wall', 'cpu_pct', 'per_core_pct', 'mem_total_gb', 'mem_used_gb', 'mem_avail_gb'}


def test_host_proc_readers_parse_this_box():
    """The /proc readers return well-formed data on a real Linux box (no psutil, procfs only)."""
    stat = _read_proc_stat()
    if stat is None:
        pytest.skip('/proc/stat unavailable on this platform')
    overall, cores = stat
    assert len(overall) == 2 and all(isinstance(v, int) for v in overall)
    assert len(cores) == os.cpu_count()  # one (idle, total) pair per core
    assert all(idle <= total for idle, total in [overall, *cores])

    total_kb, avail_kb = _read_meminfo()
    assert total_kb > 0 and 0 < avail_kb <= total_kb


def test_cpu_busy_pct_bounds_and_zero_interval():
    # Half the added jiffies were idle -> 50% busy over the interval.
    assert _cpu_busy_pct((100, 200), (150, 300)) == pytest.approx(50.0)
    # All added jiffies idle -> 0% busy; all busy -> 100%.
    assert _cpu_busy_pct((100, 200), (200, 300)) == pytest.approx(0.0)
    assert _cpu_busy_pct((100, 200), (100, 300)) == pytest.approx(100.0)
    # No elapsed jiffies (two identical snapshots) can't divide by zero.
    assert _cpu_busy_pct((100, 200), (100, 200)) == 0.0


def test_host_sampler_writes_monotone_samples_and_shuts_down_clean(tmp_path):
    sampler = _HostSampler(tmp_path, interval_s=0.05)
    if not sampler.start():
        pytest.skip('/proc/stat unavailable on this platform')
    time.sleep(0.3)
    sampler.stop()

    # Thread is joined and the file handle closed on stop() — no lingering daemon, no open fd.
    assert not sampler._thread.is_alive()
    assert sampler._fh is None

    lines = (tmp_path / HOST_LOG_FILENAME).read_text().splitlines()
    assert len(lines) >= 2, f'expected several samples at 50ms over 300ms, got {len(lines)}'

    prev_mono = None
    for line in lines:
        sample = json.loads(line)
        assert set(sample) == HOST_SAMPLE_KEYS, f'unexpected keys: {set(sample) ^ HOST_SAMPLE_KEYS}'
        assert 0.0 <= sample['cpu_pct'] <= 100.0
        assert len(sample['per_core_pct']) == os.cpu_count()
        assert all(0.0 <= pct <= 100.0 for pct in sample['per_core_pct'])
        assert sample['mem_total_gb'] > 0.0
        assert sample['mem_used_gb'] >= 0.0
        assert sample['mem_avail_gb'] >= 0.0
        assert sample['mem_used_gb'] + sample['mem_avail_gb'] == pytest.approx(sample['mem_total_gb'], abs=5e-3)
        # Monotonic timestamps strictly increase across the interval-spaced samples.
        if prev_mono is not None:
            assert sample['t_mono'] > prev_mono
        prev_mono = sample['t_mono']


def test_shim_run_lifecycle_emits_host_log(tmp_path):
    """The pass lifecycle (run/close) starts and stops the host sampler, leaving a host_stats.log behind."""
    shim = TimingShim(tmp_path, task='pick_cube', sample_gpu=False)
    with shim.run():
        if shim._host_sampler is None:
            pytest.skip('/proc/stat unavailable on this platform')
        time.sleep(0.15)
    assert shim._host_sampler is None  # stopped by close()
    assert (tmp_path / HOST_LOG_FILENAME).exists()


def test_shim_host_sampling_can_be_disabled(tmp_path):
    shim = TimingShim(tmp_path, task='pick_cube', sample_gpu=False, sample_host=False)
    with shim.run():
        assert shim._host_sampler is None
    assert not (tmp_path / HOST_LOG_FILENAME).exists()


def _write_gpu_log(out_dir: Path) -> None:
    """A minimal ``nvidia-smi dmon -s um`` shaped log so the reducer's column-driven GPU parse is exercised."""
    (out_dir / GPU_LOG_FILENAME).write_text(
        '#Date       Time        gpu    sm   mem   enc   dec    fb\n'
        '20260721 12:00:00     0    55    10     0     0  8000\n'
        '20260721 12:00:01     0    65    12     0     0  9000\n'
    )


def test_reducer_end_to_end(tmp_path):
    facts = _build_synthetic_run(tmp_path, sample_gpu=False)
    _write_gpu_log(tmp_path)  # a box without nvidia-smi has no dmon log; inject one to exercise the GPU parse
    # The host sampler's sidecar is a pass-time artifact the reducer must ignore, not join on. Drop a
    # realistic one at the dataset root and assert the reduce is unchanged by its presence.
    (tmp_path / HOST_LOG_FILENAME).write_text(
        '{"t_mono": 1.0, "t_wall": 1000.0, "cpu_pct": 12.5, "per_core_pct": [10.0, 15.0], '
        '"mem_total_gb": 31.1, "mem_used_gb": 14.2, "mem_avail_gb": 16.9}\n'
        '{"t_mono": 2.0, "t_wall": 1001.0, "cpu_pct": 33.0, "per_core_pct": [30.0, 36.0], '
        '"mem_total_gb": 31.1, "mem_used_gb": 15.0, "mem_avail_gb": 16.1}\n'
    )

    result = subprocess.run(
        ['uv', 'run', '--locked', 'positronic', 'eval', 'timing-report', '--dataset_dir', str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f'timing-report failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'

    summary = json.loads((tmp_path / 'timing_summary.json').read_text())
    assert summary['episodes'] == facts['episodes']
    assert summary['infer_calls'] == facts['episodes'] * 3
    assert summary['success_rate'] == pytest.approx(facts['expected_success_rate'])
    assert summary['real_time_factor'] > 0.0
    assert summary['infer_p50_ms'] > 0.0
    # env server decomposition was recorded, so the split is present, not null.
    assert summary['env_step_split'] is not None
    # bytes/rollout comes from the stamped size_bytes (2 MiB), joined from meta.json.
    assert summary['mean_bytes_per_rollout'] == pytest.approx(2 * 1024 * 1024)
    # the injected dmon log parses: mean sm util over {55,65}=60, peak fb 9000 MiB.
    assert summary['gpu']['sim'] is not None
    assert summary['gpu']['sim']['mean_util_pct'] == pytest.approx(60.0)
    assert summary['gpu']['sim']['peak_vram_gb'] == pytest.approx(9000 / 1024.0)
