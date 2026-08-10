import json

import pytest

from positronic.cli.eval.timing_report import (
    WallWindow,
    _build_report,
    _parse_dmon,
    _read_spans_dir,
    _read_stats_dir,
    _render,
)
from positronic.simulator.env_server.telemetry import ENV_PROCESS
from positronic.telemetry import (
    ATTR_PROCESS_NAME,
    GPU_INDEX,
    GPU_MEM_USED_B,
    GPU_PROC_MEM_B,
    GPU_UTIL_PCT,
    SPANS_SUFFIX,
    STAT_GPU_COUNT,
    STAT_GPUS,
    STAT_T_NS,
    STATS_SUFFIX,
    TELEMETRY_SUBDIR,
)
from positronic.telemetry_keys import (
    ATTR_EPISODE_ABORTED,
    ATTR_EPISODE_VIRTUAL_S,
    ATTR_PASS_VIRTUAL_CLOCK,
    HARNESS_PROCESS,
    SPAN_ENV_STEP,
    SPAN_EPISODE,
    SPAN_EVAL_PASS,
    SPAN_MATERIALIZE,
    SPAN_POLICY_INFER,
    SPAN_RECORD_IO,
    SPAN_RESET,
)

_S = 1_000_000_000  # seconds -> ns


def _span(name, start_s, end_s, span_id, parent_id=None, attrs=None, process=HARNESS_PROCESS):
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
                'resource': {'attributes': [{'key': ATTR_PROCESS_NAME, 'value': {'stringValue': process}}]},
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
    harness = [_span(SPAN_EVAL_PASS, 0, 100, 'pass0')]
    env = []
    for i, base in enumerate((0, 50)):
        ep = f'ep{i}'
        step = f'step{i}'
        harness += [
            _span(SPAN_EPISODE, base, base + 40, ep, 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
            _span(SPAN_RESET, base + 0, base + 5, f'reset{i}', ep),
            _span(SPAN_ENV_STEP, base + 10, base + 18, step, ep),
            _span(SPAN_MATERIALIZE, base + 14, base + 16, f'mat{i}', step),
            _span(SPAN_RECORD_IO, base + 20, base + 24, f'io{i}', ep),
            _span(SPAN_POLICY_INFER, base + 30, base + 33, f'inferA{i}', ep),
            _span(SPAN_POLICY_INFER, base + 33, base + 34, f'inferB{i}', ep),
        ]
        # Server env.step: 5s, decomposed into physics 3s + render 1s (server_other residual = 1s). It lives
        # in the env process's own file, discriminated by its resource process name.
        server = f'srv{i}'
        env += [
            _span(SPAN_ENV_STEP, base + 10, base + 15, server, process=ENV_PROCESS),
            _span('physics', base + 10, base + 13, f'phys{i}', server, process=ENV_PROCESS),
            _span('render', base + 13, base + 14, f'rend{i}', server, process=ENV_PROCESS),
        ]
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', harness)
    _write_lines(telemetry_dir / f'{ENV_PROCESS}{SPANS_SUFFIX}', env)
    stats = [
        {
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 50.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3}],
        },
        {
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 100.0, GPU_MEM_USED_B: 4 * 1024**3, GPU_PROC_MEM_B: 2 * 1024**3}],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))


def _wall_clock_pass(telemetry_dir):
    """One pass stamped as wall-clock — what an attended run records — with one 20 s rollout in 100 s."""
    telemetry_dir.mkdir()
    spans = [
        _span(SPAN_EVAL_PASS, 0, 100, 'pass0', attrs={ATTR_PASS_VIRTUAL_CLOCK: False}),
        _span(SPAN_EPISODE, 0, 40, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
    ]
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', spans)


def test_an_attended_pass_reports_wall_time_as_wall_time(tmp_path):
    """An attended world runs on the wall clock, so the ratio is a share of pass wall, not a real-time
    factor: reporting `sim-s per wall-s` would name a sim quantity the run never measured."""
    _wall_clock_pass(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)

    report = _build_report(spans, [], policy_gpu=None)

    assert report.virtual_clock is False
    assert report.real_time_factor is None  # there is no sim time to divide by wall time
    assert report.rollout_wall_share == pytest.approx(0.20)
    rendered = _render(report)
    assert 'rollout wall share' in rendered
    assert 'real-time factor' not in rendered
    assert 'sim-s' not in rendered


def test_a_sim_pass_still_reports_a_real_time_factor(tmp_path):
    """The virtual-clock reading is unchanged, including for a sidecar written before the pass carried
    its clock — every one of those is a sim sweep."""
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)

    report = _build_report(spans, _read_stats_dir(tmp_path / TELEMETRY_SUBDIR), policy_gpu=None)

    assert report.virtual_clock is True  # the fixture's pass carries no clock attr
    assert report.real_time_factor == pytest.approx(0.40)
    assert 'real-time factor' in _render(report)


def test_report_aggregates(tmp_path):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(spans, _read_stats_dir(tmp_path / TELEMETRY_SUBDIR), policy_gpu=None)

    assert report.episodes == 2
    assert report.window is WallWindow.W_PASS
    assert report.wall_s == pytest.approx(100.0)
    assert report.real_time_factor == pytest.approx(0.40)  # 40 virtual-s / 100 wall-s
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


def test_render_shows_shares_as_percentages(tmp_path):
    """Every unitless share renders as an aligned percentage, and the policy-wait row carries the serving
    capacity derived from it."""
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(spans, _read_stats_dir(tmp_path / TELEMETRY_SUBDIR), policy_gpu=None)

    rendered = _render(report).splitlines()

    assert 'real-time factor:      40.0% (sim-s per wall-s)' in rendered
    assert '  policy_wait         8.0%  -> ~12.5 sims per policy server' in rendered
    assert '  between_episodes   20.0%' in rendered  # the longest label still lands in the same column
    assert '  physics            37.5%' in rendered
    assert '  materialize        25.0%' in rendered


def test_render_omits_serving_capacity_without_inference(tmp_path):
    """A pass with no inference has a zero policy-wait share, which yields no sims-per-server figure."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
            _span(SPAN_EPISODE, 0, 40, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        ],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    rendered = _render(report).splitlines()

    assert '  policy_wait         0.0%' in rendered
    assert not any('sims per policy server' in line for line in rendered)


def test_aborted_episode_excluded(tmp_path):
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    docs = [
        _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
        _span(SPAN_EPISODE, 0, 40, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        _span(SPAN_EPISODE, 50, 60, 'ep1', 'pass0'),  # aborted: dropped from the count and the reduce
    ]
    docs[2]['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'].append({
        'key': ATTR_EPISODE_ABORTED,
        'value': {'boolValue': True},
    })
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', docs)
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.episodes == 1


def test_aborted_episode_wall_excluded_from_between_episodes(tmp_path):
    """An aborted rollout is dropped as invalid data, so its wall must leave W_pass entirely — not fall into
    ``between_episodes`` and not deflate the policy-busy / real-time factors. Here a 40 s aborted episode sits
    beside a 40 s completed one in a 100 s pass, so the valid W_pass is 60 s."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    harness = [
        _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
        _span(SPAN_EPISODE, 0, 40, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        _span(SPAN_RESET, 0, 5, 'reset0', 'ep0'),
        _span(SPAN_ENV_STEP, 10, 18, 'step0', 'ep0'),
        _span(SPAN_MATERIALIZE, 14, 16, 'mat0', 'step0'),
        _span(SPAN_RECORD_IO, 20, 24, 'io0', 'ep0'),
        _span(SPAN_POLICY_INFER, 30, 33, 'inferA0', 'ep0'),
        _span(SPAN_POLICY_INFER, 33, 34, 'inferB0', 'ep0'),
        _span(SPAN_EPISODE, 50, 90, 'ep1', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),  # 40 s of real wall, aborted
    ]
    harness[-1]['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'].append({
        'key': ATTR_EPISODE_ABORTED,
        'value': {'boolValue': True},
    })
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', harness)

    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    assert report.episodes == 1
    assert report.wall_s == pytest.approx(60.0)  # 100 s pass minus the 40 s aborted episode
    split = report.wall_split
    # between_episodes is the completed episode's real inter-episode idle only (60 - 40), NOT (100 - 40)/100
    # with the aborted wall folded in.
    assert split.between_episodes == pytest.approx(20 / 60)
    assert split.reset == pytest.approx(5 / 60)
    assert split.env_step == pytest.approx(8 / 60)
    assert split.record_io == pytest.approx(4 / 60)
    assert split.overhead == pytest.approx(19 / 60)
    assert sum(vars(split).values()) == pytest.approx(1.0)
    # 60 s is the denominator of these figures too — the aborted wall is out of W_pass, not merely unattributed.
    assert split.policy_wait == pytest.approx(4 / 60)
    assert report.real_time_factor == pytest.approx(20 / 60)


def test_env_step_split_ignores_aborted_episode(tmp_path):
    """An aborted rollout's spans — its client env.step AND its server-side steps — stay out of the env-step
    split: the denominator covers completed episodes only, so counting them would push the fractions past 1 and
    the wire residual below 0."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    harness = [
        _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
        _span(SPAN_EPISODE, 0, 40, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        _span(SPAN_ENV_STEP, 10, 18, 'step0', 'ep0'),
        _span(SPAN_MATERIALIZE, 14, 16, 'mat0', 'step0'),
        _span(SPAN_EPISODE, 50, 60, 'ep1', 'pass0'),
        _span(SPAN_ENV_STEP, 51, 59, 'step1', 'ep1'),  # aborted episode's client step
    ]
    harness[4]['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'].append({
        'key': ATTR_EPISODE_ABORTED,
        'value': {'boolValue': True},
    })
    env = [
        _span(SPAN_ENV_STEP, 10, 15, 'srv0', process=ENV_PROCESS),
        _span('physics', 10, 13, 'phys0', 'srv0', process=ENV_PROCESS),
        _span('render', 13, 14, 'rend0', 'srv0', process=ENV_PROCESS),
        _span(SPAN_ENV_STEP, 51, 56, 'srv1', process=ENV_PROCESS),  # during the aborted rollout
        _span('physics', 51, 54, 'phys1', 'srv1', process=ENV_PROCESS),
    ]
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', harness)
    _write_lines(telemetry_dir / f'{ENV_PROCESS}{SPANS_SUFFIX}', env)

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
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, 0, 30, 'ghost-ep', 'ghost-pass', {ATTR_EPISODE_VIRTUAL_S: 15.0}),  # killed earlier run
            _span(SPAN_EVAL_PASS, 100, 200, 'pass0'),
            _span(SPAN_EPISODE, 100, 140, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        ],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.episodes == 1
    assert report.window is WallWindow.W_PASS
    assert report.real_time_factor == pytest.approx(0.20)  # 20 virtual-s / 100 wall-s; the orphan's 15 don't count
    assert report.wall_split.between_episodes == pytest.approx(0.60)


def test_run_whose_pass_span_never_closed_reduces_over_its_episodes(tmp_path):
    """A killed or preempted run writes no ``eval.pass`` span, but its finished episodes are complete recorded
    data. They reduce against the wall they span — 10 s to 100 s here — which the report names W_episodes
    because it excludes whatever ran either side of them and is therefore not a pass window."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, 10, 50, 'ep0', 'unwritten-pass', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
            _span(SPAN_RESET, 10, 15, 'reset0', 'ep0'),
            _span(SPAN_EPISODE, 60, 100, 'ep1', 'unwritten-pass', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        ],
    )

    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    assert report.episodes == 2
    assert report.window is WallWindow.W_EPISODES
    assert report.wall_s == pytest.approx(90.0)
    assert report.real_time_factor == pytest.approx(40 / 90)
    assert report.wall_split.reset == pytest.approx(5 / 90)
    # The 10 s between the two episodes is inside the window and attributes; the 10 s before the first is not.
    assert report.wall_split.between_episodes == pytest.approx(10 / 90)


def test_two_killed_runs_in_one_directory_get_a_window_each(tmp_path):
    """Grouping episodes by the pass span they name keeps two killed runs appended to one directory apart. The
    idle wall between them belongs to neither run, exactly as the gap between two pass spans does — one window
    spanning both would report 1040 s of wall for 80 s of work."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, 0, 40, 'ep0', 'unwritten-pass-a', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
            _span(SPAN_EPISODE, 1000, 1040, 'ep1', 'unwritten-pass-b', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        ],
    )

    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    assert report.episodes == 2
    assert report.wall_s == pytest.approx(80.0)
    assert report.wall_split.between_episodes == pytest.approx(0.0)


def test_telemetry_with_neither_a_pass_nor_an_episode_names_what_is_missing(tmp_path):
    """Spans that carry no closed pass and no episode leave nothing to reduce. The refusal must say so: a run
    killed before its first episode finished did record telemetry, so blaming a missing ``--timing`` sends the
    reader after a flag they set."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_RESET, 0, 5, 'reset0')])

    with pytest.raises(ValueError, match=f'no closed `{SPAN_EVAL_PASS}` span and no `{SPAN_EPISODE}` span') as exc:
        _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert '--timing' not in str(exc.value)


def test_render_names_the_episode_window_it_reduced_over(tmp_path):
    """The console warning is lost once the report is pasted elsewhere, so the rendered body itself must say
    the shares are of the episode window rather than of a pass."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [_span(SPAN_EPISODE, 0, 90, 'ep0', 'unwritten-pass', {ATTR_EPISODE_VIRTUAL_S: 20.0})],
    )

    rendered = _render(_build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)).splitlines()

    assert f'no {SPAN_EVAL_PASS} span closed' in rendered[0]
    assert f'{WallWindow.W_EPISODES} (wall):   90.0 s (0.03 h)' in rendered
    assert f'wall split (share of {WallWindow.W_EPISODES}):' in rendered
    assert not any(WallWindow.W_PASS in line for line in rendered)


def test_multi_gpu_peak_vram_sums_devices_per_sample(tmp_path):
    """Peak VRAM on a multi-GPU box is the box-wide total at one instant — each sample's devices summed, then
    the max over samples — not the largest single-device reading."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        {
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 60.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: None},
            ],
        },
        {
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 80.0, GPU_MEM_USED_B: 1 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 20.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: None},
            ],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.mean_util_pct == pytest.approx(50.0)  # mean over the four per-GPU readings
    assert sim.peak_vram_gb == pytest.approx(5.0)  # sample 1's 2+3 GB total, not device 1's 3 GB
    # Every device answered, so the mean covers the box and the rendered line carries no coverage qualifier.
    assert (sim.devices_seen, sim.box_devices) == (2, 2)
    assert 'util 50%  peak VRAM' in _render(report)
    # Every sample has a GPU with no per-process attribution, so no sample can carry a box-wide process total.
    assert sim.peak_proc_vram_gb is None


def test_partial_per_gpu_proc_vram_excluded_from_peak(tmp_path):
    """A sample where some GPU can't attribute process memory (``proc_mem_b`` None) is incomplete for the
    box-wide process-VRAM peak and contributes nothing; the peak reflects only samples where every GPU
    reported. The sample still carries every device, so it counts towards util and the box-wide mem peak."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        {  # incomplete: gpu1 unattributed -> must not add gpu0's 5 GB alone to the peak
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 5 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 60.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: None},
            ],
        },
        {  # complete: box-wide process VRAM is 1+2 = 3 GB
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 80.0, GPU_MEM_USED_B: 1 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 20.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: 2 * 1024**3},
            ],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.peak_proc_vram_gb == pytest.approx(3.0)  # complete sample only; the incomplete 5 GB is dropped
    assert sim.peak_vram_gb == pytest.approx(5.0)  # box-wide mem peak unaffected: sample 1's 2+3 GB


def test_omitted_gpu_device_excluded_from_proc_vram_peak(tmp_path):
    """A device whose NVML query errors mid-run is dropped from the sample, so it carries fewer GPUs than the
    box holds. Such a sample is incomplete for the box-wide process peak even though every GPU it *does* carry
    reported ``proc_mem_b`` — its large lone reading must not win over a complete sample."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        {  # complete: both GPUs present, matching the recorded complement of 2, for a 1+2 = 3 GB total
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 60.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: 2 * 1024**3},
            ],
        },
        {  # gpu1 omitted (errored mid-run): only one device against a recorded count of 2, so incomplete
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 80.0, GPU_MEM_USED_B: 1 * 1024**3, GPU_PROC_MEM_B: 9 * 1024**3}],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    # Peak reflects only the complete two-device sample; the omitted-device sample's 9 GB is dropped. Asking
    # whether every device PRESENT reported would pass over the single one and let the 9 GB win.
    assert sim.peak_proc_vram_gb == pytest.approx(3.0)


def test_device_omitted_from_every_sample_excluded_via_recorded_count(tmp_path):
    """When a device is omitted from EVERY in-window sample, no sample carries the box's full complement, so
    max-observed would wrongly infer the complement from the surviving device and count every sample complete.
    The recorded ``gpu_count`` keeps the true count (2), so the one-device samples are incomplete and neither
    box-wide peak can be reported. Mean utilisation still covers the device that answered."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        # 2-GPU box, but device 1 is missing from both samples; each records the configured count of 2.
        {
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 5 * 1024**3}],
        },
        {
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 80.0, GPU_MEM_USED_B: 1 * 1024**3, GPU_PROC_MEM_B: 7 * 1024**3}],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    # No sample carries both GPUs, so none is complete: the peak is ``None``. Inferring the complement from the
    # devices observed would read 1, count both single-device samples complete, and report a 7 GB peak.
    assert sim.peak_proc_vram_gb is None
    # Peak VRAM is a box-wide sum too, so an incomplete sample would report half the box as the whole of it.
    assert sim.peak_vram_gb is None
    # The mean still averages the surviving device's readings, and carries the coverage that biases it.
    assert sim.mean_util_pct == pytest.approx(60.0)
    assert (sim.devices_seen, sim.box_devices) == (1, 2)
    assert 'util 60% over 1 of 2 GPUs' in _render(report)


def test_sample_of_a_smaller_complement_cannot_set_the_box_peak(tmp_path):
    """An output directory resumed on a box with fewer GPUs holds samples of two complements. The summary
    reports the larger one, so a smaller complement's totals cover part of that box and must not become its
    peak, complete though they are within their own run."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        {  # a 2-GPU box, whole: 1+2 = 3 GB of process VRAM
            STAT_T_NS: 1,
            STAT_GPU_COUNT: 2,
            STAT_GPUS: [
                {GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3, GPU_PROC_MEM_B: 1 * 1024**3},
                {GPU_INDEX: 1, GPU_UTIL_PCT: 60.0, GPU_MEM_USED_B: 3 * 1024**3, GPU_PROC_MEM_B: 2 * 1024**3},
            ],
        },
        {  # a later run on a 1-GPU box: complete for its own complement, half a box for this summary
            STAT_T_NS: 2,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 80.0, GPU_MEM_USED_B: 9 * 1024**3, GPU_PROC_MEM_B: 8 * 1024**3}],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.box_devices == 2
    assert sim.peak_proc_vram_gb == pytest.approx(3.0)  # the two-device sample, not the 1-GPU run's 8 GB
    assert sim.peak_vram_gb == pytest.approx(5.0)  # likewise 2+3 GB, not the 1-GPU run's 9 GB
    # The mean is a figure about this box too: the other machine's 80% reading is not averaged into it.
    assert sim.mean_util_pct == pytest.approx(50.0)


def test_box_whose_devices_all_refuse_is_reported_with_metrics_unavailable(tmp_path):
    """A box whose every device refuses its query (a single unsupported MIG device is enough) records a
    positive ``gpu_count`` with no per-device entries. That is a GPU box with nothing measured, and reporting
    it as CPU-only — no GPU line at all — hides both the box and the failure."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [{STAT_T_NS: 1, STAT_GPU_COUNT: 2, STAT_GPUS: []}, {STAT_T_NS: 2, STAT_GPU_COUNT: 2, STAT_GPUS: []}]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.mean_util_pct is None
    assert (sim.devices_seen, sim.box_devices) == (0, 2)
    assert sim.peak_vram_gb is None
    assert sim.peak_proc_vram_gb is None
    assert 'util unavailable over 0 of 2 GPUs' in _render(report)


def test_cpu_box_has_no_gpu_summary(tmp_path):
    """A box NVML found no device on records a zero ``gpu_count``, and carries no GPU summary at all."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(
        json.dumps({STAT_T_NS: 1, STAT_GPU_COUNT: 0, STAT_GPUS: []}) + '\n'
    )

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    assert report.gpu.sim is None
    assert 'gpu[sim]' not in _render(report)


def test_gpu_samples_outside_pass_windows_excluded(tmp_path):
    """Stats samples taken outside every completed pass's wall window (an earlier run in a reused directory)
    stay out of the GPU summary — the stats twin of the orphan-episode exclusion."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 100, 200, 'pass0')])
    stats = [
        {
            STAT_T_NS: 50 * _S,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 100.0, GPU_MEM_USED_B: 8 * 1024**3}],
        },  # earlier run
        {
            STAT_T_NS: 150 * _S,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: 40.0, GPU_MEM_USED_B: 2 * 1024**3}],
        },
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.mean_util_pct == pytest.approx(40.0)  # the 100% sample predates the pass
    assert sim.peak_vram_gb == pytest.approx(2.0)


def test_native_sim_has_no_env_step_split(tmp_path):
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    docs = [
        _span(SPAN_EVAL_PASS, 0, 10, 'pass0'),
        _span(SPAN_EPISODE, 0, 8, 'ep0', 'pass0', {ATTR_EPISODE_VIRTUAL_S: 4.0}),
        _span(SPAN_ENV_STEP, 1, 3, 'step0', 'ep0'),  # client only; no server env.step in the file
    ]
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', docs)
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert report.env_step_split is None  # no env server reported a decomposition


def test_parse_dmon_reads_sm_and_fb(tmp_path):
    log = tmp_path / 'dmon.log'
    log.write_text('# gpu    sm    fb\n#  Idx     %    MB\n    0    50  1024\n    0   100  2048\n')
    summary = _parse_dmon(log)
    assert summary.mean_util_pct == pytest.approx(75.0)
    assert summary.peak_vram_gb == pytest.approx(2048 / 1024)
    assert summary.peak_proc_vram_gb is None


def test_parse_dmon_multi_gpu_sums_devices_per_cycle(tmp_path):
    """A multi-GPU dmon log groups device rows into sampling cycles (the gpu index repeating opens the next),
    and peak VRAM is the max over per-cycle device sums — the box total at one instant, not the largest single
    device row."""
    log = tmp_path / 'dmon.log'
    log.write_text(
        '# gpu    sm    fb\n'
        '#  Idx     %    MB\n'
        '    0    50  1024\n'
        '    1    50  3072\n'  # cycle 1 total: 4096
        '    0   100  2048\n'
        '    1   100  1024\n'  # cycle 2 total: 3072
    )
    summary = _parse_dmon(log)
    assert summary.mean_util_pct == pytest.approx(75.0)
    assert summary.peak_vram_gb == pytest.approx(4096 / 1024)  # cycle 1's sum, not device 1's 3072 alone


def test_parse_dmon_unreadable_row_still_closes_its_cycle(tmp_path):
    """A row whose metrics are non-numeric (``-`` on a device that refused the query) contributes no reading
    but still marks its device in the open cycle. Dropping it whole would leave the cycle unclosed, so the
    next interval's device-0 reading would sum into this interval's box total."""
    log = tmp_path / 'dmon.log'
    log.write_text(
        '# gpu    sm    fb\n'
        '#  Idx     %    MB\n'
        '    0     -     -\n'  # cycle 1: device 0 refused the query
        '    1    50  6144\n'  # cycle 1 total: 6144
        '    0   100  8192\n'
        '    1   100  1024\n'  # cycle 2 total: 9216
    )
    summary = _parse_dmon(log)
    # Without the boundary, cycle 1's device 1 (6144) and cycle 2's device 0 (8192) sum into one 14336 MB
    # "instant" the box never held, and cycle 2 is left with device 1 alone.
    assert summary.peak_vram_gb == pytest.approx(9216 / 1024)
    assert summary.mean_util_pct == pytest.approx(250 / 3)  # the three readable rows: 50, 100, 100


def test_parse_dmon_partial_cycle_excluded_from_peak(tmp_path):
    """The peak is a box-wide total at one instant, so an interval missing a device is not a candidate:
    one device's 12 GiB with its neighbour unread would otherwise outrank a complete 11 GiB instant and be
    reported as the box peak."""
    log = tmp_path / 'dmon.log'
    log.write_text(
        '# gpu    sm    fb\n'
        '#  Idx     %    MB\n'
        '    0     -     -\n'
        '    1    50 12288\n'  # partial: 12 GiB, device 0 unread
        '    0   100  5120\n'
        '    1   100  6144\n'  # complete: 11 GiB
    )
    summary = _parse_dmon(log)
    assert summary.peak_vram_gb == pytest.approx(11264 / 1024)
    assert (summary.devices_seen, summary.box_devices) == (2, 2)


def test_parse_dmon_device_unreadable_throughout_counts_towards_the_box(tmp_path):
    """A device that prints `-` in every interval still appears in the log, so it is part of the box the
    mean is measured against — and no interval ever covers that box, so there is no peak to report."""
    log = tmp_path / 'dmon.log'
    log.write_text(
        '# gpu    sm    fb\n'
        '#  Idx     %    MB\n'
        '    0     -     -\n'
        '    1    50  1024\n'
        '    0     -     -\n'
        '    1    70  2048\n'
    )
    summary = _parse_dmon(log)
    assert (summary.devices_seen, summary.box_devices) == (1, 2)
    assert summary.peak_vram_gb is None
    assert summary.mean_util_pct == pytest.approx(60.0)


def test_parse_dmon_reads_each_metric_on_its_own(tmp_path):
    """A device can report one metric and not the other — dmon prints `-` per column. A row carrying
    utilisation but no framebuffer still contributes its utilisation, and one carrying a framebuffer but no
    utilisation still contributes to the box total."""
    log = tmp_path / 'dmon.log'
    log.write_text(
        '# gpu    sm    fb\n'
        '#  Idx     %    MB\n'
        '    0    40     -\n'  # utilisation only
        '    1     -  2048\n'  # framebuffer only
        '    0    80  1024\n'
        '    1   100  2048\n'
    )
    summary = _parse_dmon(log)
    # Cycle 1 holds device 1 alone, so it is partial; cycle 2 sees both.
    assert summary.peak_vram_gb == pytest.approx(3072 / 1024)
    assert summary.mean_util_pct == pytest.approx(220 / 3)  # 40, 80, 100 — device 1's `-` costs only itself
    assert (summary.devices_seen, summary.box_devices) == (2, 2)


def test_parse_dmon_fails_loudly_without_fb(tmp_path):
    log = tmp_path / 'dmon.log'
    log.write_text('# gpu    sm\n#  Idx     %\n    0    50\n')
    with pytest.raises(ValueError, match='fb'):
        _parse_dmon(log)
