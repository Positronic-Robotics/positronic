import json

import pytest

from positronic.cli.eval.timing_report import _build_report, _parse_dmon, _read_spans_dir, _read_stats_dir

_S = 1_000_000_000  # seconds -> ns


def _span(name, start_s, end_s, span_id, parent_id=None, attrs=None, process='harness'):
    encoded = {
        'traceId': '0' * 32,
        'spanId': span_id,
        'name': name,
        'startTimeUnixNano': str(int(start_s * _S)),
        'endTimeUnixNano': str(int(end_s * _S)),
        'attributes': [{'key': k, 'value': {'doubleValue': v}} for k, v in (attrs or {}).items()],
    }
    if parent_id is not None:
        encoded['parentSpanId'] = parent_id
    return {
        'resourceSpans': [
            {
                'resource': {'attributes': [{'key': 'process.name', 'value': {'stringValue': process}}]},
                'scopeSpans': [{'spans': [encoded]}],
            }
        ]
    }


def _write_lines(path, docs):
    path.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))


def _fixture(telemetry_dir):
    """One pass, two identical episodes, each with reset/env.step(+materialize)/record.io/two infers, plus a
    server env.step (physics/render) per episode in the env file, and a two-sample GPU stats stream."""
    telemetry_dir.mkdir()
    harness = [_span('eval.pass', 0, 100, 'pass0')]
    env = []
    for i, base in enumerate((0, 50)):
        ep = f'ep{i}'
        step = f'step{i}'
        harness += [
            _span('episode', base, base + 40, ep, 'pass0', {'episode.virtual_s': 20.0}),
            _span('reset', base + 0, base + 5, f'reset{i}', ep),
            _span('env.step', base + 10, base + 18, step, ep),
            _span('materialize', base + 14, base + 16, f'mat{i}', step),
            _span('record.io', base + 20, base + 24, f'io{i}', ep),
            _span('policy.infer', base + 30, base + 33, f'inferA{i}', ep),
            _span('policy.infer', base + 33, base + 34, f'inferB{i}', ep),
        ]
        # Server env.step: 5s, decomposed into physics 3s + render 1s (server_other residual = 1s). It lives
        # in the env process's own file, discriminated by its resource process name.
        server = f'srv{i}'
        env += [
            _span('env.step', base + 10, base + 15, server, process='env'),
            _span('physics', base + 10, base + 13, f'phys{i}', server, process='env'),
            _span('render', base + 13, base + 14, f'rend{i}', server, process='env'),
        ]
    _write_lines(telemetry_dir / 'harness.spans.jsonl', harness)
    _write_lines(telemetry_dir / 'env.spans.jsonl', env)
    stats = [
        {'t_ns': 1, 'gpus': [{'i': 0, 'util_pct': 50.0, 'mem_used_b': 2 * 1024**3, 'proc_mem_b': 1 * 1024**3}]},
        {'t_ns': 2, 'gpus': [{'i': 0, 'util_pct': 100.0, 'mem_used_b': 4 * 1024**3, 'proc_mem_b': 2 * 1024**3}]},
    ]
    (telemetry_dir / 'harness.stats.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in stats))


def test_report_aggregates(tmp_path):
    _fixture(tmp_path / 'telemetry')
    spans = _read_spans_dir(tmp_path / 'telemetry')
    report = _build_report(spans, _read_stats_dir(tmp_path / 'telemetry'), policy_gpu=None)

    assert report.episodes == 2
    assert report.wall_pass_s == pytest.approx(100.0)
    assert report.real_time_factor == pytest.approx(0.40)  # 40 virtual-s / 100 wall-s
    assert report.policy_busy_fraction == pytest.approx(0.08)  # 8 infer-s / 100
    assert report.infer_calls == 4
    assert report.infer_p50_ms == pytest.approx(2000.0)
    assert report.infer_p95_ms == pytest.approx(3000.0)

    split = report.wall_split
    assert split.reset == pytest.approx(0.10)
    assert split.env_step == pytest.approx(0.16)  # includes materialize
    assert split.policy_wait == pytest.approx(0.08)
    assert split.record_io == pytest.approx(0.08)
    assert split.overhead == pytest.approx(0.38)
    assert split.between_episodes == pytest.approx(0.20)
    assert sum(vars(split).values()) == pytest.approx(1.0)

    assert report.env_step_split is not None
    env_split = report.env_step_split
    assert env_split.phases['physics'] == pytest.approx(6 / 16)
    assert env_split.phases['render'] == pytest.approx(2 / 16)
    assert env_split.phases['server_other'] == pytest.approx(2 / 16)
    assert env_split.materialize == pytest.approx(4 / 16)
    assert env_split.wire == pytest.approx(2 / 16)
    assert sum(env_split.phases.values()) + env_split.wire + env_split.materialize == pytest.approx(1.0)

    assert report.gpu.sim is not None
    assert report.gpu.sim.mean_util_pct == pytest.approx(75.0)
    assert report.gpu.sim.peak_vram_gb == pytest.approx(4.0)
    assert report.gpu.sim.peak_proc_vram_gb == pytest.approx(2.0)
    assert report.gpu.policy is None


def test_aborted_episode_excluded(tmp_path):
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    docs = [
        _span('eval.pass', 0, 100, 'pass0'),
        _span('episode', 0, 40, 'ep0', 'pass0', {'episode.virtual_s': 20.0}),
        _span('episode', 50, 60, 'ep1', 'pass0'),  # aborted: dropped from the count and the reduce
    ]
    docs[2]['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'].append({
        'key': 'episode.aborted',
        'value': {'boolValue': True},
    })
    _write_lines(telemetry_dir / 'harness.spans.jsonl', docs)
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.episodes == 1


def test_env_step_split_ignores_aborted_episode(tmp_path):
    """An aborted rollout's spans — its client env.step AND its server-side steps — stay out of the env-step
    split: the denominator covers completed episodes only, so counting them made fractions exceed 1 and wire
    go negative (Codex P2 on PR #531)."""
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    harness = [
        _span('eval.pass', 0, 100, 'pass0'),
        _span('episode', 0, 40, 'ep0', 'pass0', {'episode.virtual_s': 20.0}),
        _span('env.step', 10, 18, 'step0', 'ep0'),
        _span('materialize', 14, 16, 'mat0', 'step0'),
        _span('episode', 50, 60, 'ep1', 'pass0'),
        _span('env.step', 51, 59, 'step1', 'ep1'),  # aborted episode's client step
    ]
    harness[4]['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'].append({
        'key': 'episode.aborted',
        'value': {'boolValue': True},
    })
    env = [
        _span('env.step', 10, 15, 'srv0', process='env'),
        _span('physics', 10, 13, 'phys0', 'srv0', process='env'),
        _span('render', 13, 14, 'rend0', 'srv0', process='env'),
        _span('env.step', 51, 56, 'srv1', process='env'),  # during the aborted rollout
        _span('physics', 51, 54, 'phys1', 'srv1', process='env'),
    ]
    _write_lines(telemetry_dir / 'harness.spans.jsonl', harness)
    _write_lines(telemetry_dir / 'env.spans.jsonl', env)

    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    split = report.env_step_split
    assert split is not None
    # Only the completed episode's 8s client step and its 5s server step count: physics 3/8, render 1/8,
    # server_other 1/8, materialize 2/8, wire (8-5-2)/8.
    assert split.phases['physics'] == pytest.approx(3 / 8)
    assert split.phases['render'] == pytest.approx(1 / 8)
    assert split.phases['server_other'] == pytest.approx(1 / 8)
    assert split.materialize == pytest.approx(2 / 8)
    assert split.wire == pytest.approx(1 / 8)
    assert sum(split.phases.values()) + split.wire + split.materialize == pytest.approx(1.0)


def test_orphan_episode_from_killed_run_excluded(tmp_path):
    """A killed run flushes its episodes but never writes its ``eval.pass`` span; when the directory is reused,
    those orphans must not reduce — they would inflate every pass-normalized figure."""
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / 'harness.spans.jsonl',
        [
            _span('episode', 0, 30, 'ghost-ep', 'ghost-pass', {'episode.virtual_s': 15.0}),  # killed earlier run
            _span('eval.pass', 100, 200, 'pass0'),
            _span('episode', 100, 140, 'ep0', 'pass0', {'episode.virtual_s': 20.0}),
        ],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.episodes == 1
    assert report.real_time_factor == pytest.approx(0.20)  # 20 virtual-s / 100 wall-s; the orphan's 15 don't count
    assert report.wall_split.between_episodes == pytest.approx(0.60)


def test_multi_gpu_peak_vram_sums_devices_per_sample(tmp_path):
    """Peak VRAM on a multi-GPU box is the box-wide total at one instant — each sample's devices summed, then
    the max over samples — not the largest single-device reading."""
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / 'harness.spans.jsonl', [_span('eval.pass', 0, 10, 'pass0')])
    stats = [
        {
            't_ns': 1,
            'gpus': [
                {'i': 0, 'util_pct': 40.0, 'mem_used_b': 2 * 1024**3, 'proc_mem_b': 1 * 1024**3},
                {'i': 1, 'util_pct': 60.0, 'mem_used_b': 3 * 1024**3, 'proc_mem_b': None},
            ],
        },
        {
            't_ns': 2,
            'gpus': [
                {'i': 0, 'util_pct': 80.0, 'mem_used_b': 1 * 1024**3, 'proc_mem_b': 1 * 1024**3},
                {'i': 1, 'util_pct': 20.0, 'mem_used_b': 3 * 1024**3, 'proc_mem_b': None},
            ],
        },
    ]
    (telemetry_dir / 'harness.stats.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.mean_util_pct == pytest.approx(50.0)  # mean over the four per-GPU readings
    assert sim.peak_vram_gb == pytest.approx(5.0)  # sample 1's 2+3 GB total, not device 1's 3 GB
    assert sim.peak_proc_vram_gb == pytest.approx(1.0)  # only attributed devices count


def test_native_sim_has_no_env_step_split(tmp_path):
    telemetry_dir = tmp_path / 'telemetry'
    telemetry_dir.mkdir()
    docs = [
        _span('eval.pass', 0, 10, 'pass0'),
        _span('episode', 0, 8, 'ep0', 'pass0', {'episode.virtual_s': 4.0}),
        _span('env.step', 1, 3, 'step0', 'ep0'),  # client only; no server env.step in the file
    ]
    _write_lines(telemetry_dir / 'harness.spans.jsonl', docs)
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.env_step_split is None  # no env server reported a decomposition


def test_parse_dmon_reads_sm_and_fb(tmp_path):
    log = tmp_path / 'dmon.log'
    log.write_text('# gpu    sm    fb\n#  Idx     %    MB\n    0    50  1024\n    0   100  2048\n')
    summary = _parse_dmon(log)
    assert summary.mean_util_pct == pytest.approx(75.0)
    assert summary.peak_vram_gb == pytest.approx(2048 / 1024)
    assert summary.peak_proc_vram_gb is None


def test_parse_dmon_fails_loudly_without_fb(tmp_path):
    log = tmp_path / 'dmon.log'
    log.write_text('# gpu    sm\n#  Idx     %\n    0    50\n')
    with pytest.raises(ValueError, match='fb'):
        _parse_dmon(log)
