import json
import re

import pos3
import pyarrow as pa
import pytest
from positronic_model_server import protocol

from positronic.cli import _internal_main
from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.eval.timing_report import (
    WallWindow,
    _build_report,
    _parse_dmon,
    _read_spans_dir,
    _read_stats_dir,
    _render,
    timing_report,
)
from positronic.simulator.env_server.telemetry import ENV_PROCESS
from positronic.telemetry import (
    ATTR_PROCESS_NAME,
    ATTR_RUN_ID,
    COMPRESSED_SPANS_SUFFIX,
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
    read_spans,
)
from positronic.telemetry_keys import (
    ATTR_CODEC,
    ATTR_EPISODE_VIRTUAL_S,
    ATTR_SERVED_PREFIX,
    ATTR_STEP_GC_GENERATION,
    ATTR_STEP_GC_MS,
    ATTR_STEP_LATE_MS,
    ATTR_STEP_POLICY_MS,
    ATTR_STEP_RECORD_WAIT_MS,
    ATTR_WAYPOINTS_DROPPED,
    ATTR_WAYPOINTS_EMITTED,
    ATTR_WAYPOINTS_LATE_MAX_MS,
    ATTR_WAYPOINTS_LATE_SUM_MS,
    ATTR_WAYPOINTS_SCHEDULED,
    ATTR_WIRE_DOWNLINK_MS,
    ATTR_WIRE_NETWORK_MS,
    ATTR_WIRE_SENT_ALONE,
    ATTR_WIRE_UPLINK_MS,
    HARNESS_PROCESS,
    SPAN_ENV_STEP,
    SPAN_EPISODE,
    SPAN_EVAL_PASS,
    SPAN_HARNESS_STEP,
    SPAN_MATERIALIZE,
    SPAN_POLICY_ENCODE,
    SPAN_POLICY_INFER,
    SPAN_POLICY_PREPARE,
    SPAN_POLICY_SUBMIT,
    SPAN_RECORD_IO,
    SPAN_RESET,
    SPAN_WIRE_RECV,
    SPAN_WIRE_SEND,
)

_S = 1_000_000_000  # seconds -> ns


def _span(name, start_s, end_s, span_id, parent_id=None, attrs=None, process=HARNESS_PROCESS, run_id=''):
    encoded = {
        'traceId': '0' * 32,
        'spanId': span_id,
        'name': name,
        'startTimeUnixNano': str(int(start_s * _S)),
        'endTimeUnixNano': str(int(end_s * _S)),
        'attributes': [
            {'key': k, 'value': {'stringValue': v} if isinstance(v, str) else {'doubleValue': v}}
            for k, v in (attrs or {}).items()
        ],
    }
    if parent_id is not None:
        encoded['parentSpanId'] = parent_id
    resource = [{'key': ATTR_PROCESS_NAME, 'value': {'stringValue': process}}]
    if run_id:
        resource.append({'key': ATTR_RUN_ID, 'value': {'stringValue': run_id}})
    return {'resourceSpans': [{'resource': {'attributes': resource}, 'scopeSpans': [{'spans': [encoded]}]}]}


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


def test_a_zstd_span_file_reads_as_the_plain_file_it_holds(tmp_path):
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    _fixture(telemetry_dir)
    plain = _read_spans_dir(telemetry_dir)
    harness = telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}'
    frame = pa.Codec('zstd', compression_level=3).compress(harness.read_bytes(), asbytes=True)
    (telemetry_dir / f'{HARNESS_PROCESS}{COMPRESSED_SPANS_SUFFIX}').write_bytes(frame)
    harness.unlink()

    assert sorted(_read_spans_dir(telemetry_dir)) == sorted(plain)


@pytest.mark.parametrize('nested_inference', [False, True])
def test_report_aggregates(tmp_path, nested_inference):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)
    if nested_inference:
        nested = []
        for span in spans:
            if span.name == SPAN_POLICY_INFER:
                call = span._replace(name='processor', span_id=f'call-{span.span_id}', end_ns=span.start_ns + 1)
                worker = span._replace(name=SPAN_POLICY_SUBMIT, span_id=f'work-{span.span_id}', parent_id=call.span_id)
                nested.extend([call, worker, span._replace(parent_id=worker.span_id)])
            else:
                nested.append(span)
        spans = nested
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


def test_the_record_io_share_counts_the_recording_between_steps_that_the_step_spans_carry(tmp_path):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    spans = _read_spans_dir(tmp_path / TELEMETRY_SUBDIR)
    steps = [
        _span(SPAN_HARNESS_STEP, base + 35, base + 36, f'hstep{i}', f'ep{i}', {ATTR_STEP_RECORD_WAIT_MS: 2000.0})
        for i, base in enumerate((0, 50))
    ]
    _write_lines(tmp_path / 'steps.jsonl', steps)
    spans += list(read_spans(tmp_path / 'steps.jsonl'))
    split = _build_report(spans, _read_stats_dir(tmp_path / TELEMETRY_SUBDIR), policy_gpu=None).wall_split

    assert split.record_io == pytest.approx(0.12)  # 4 s of record.io spans and 2 s between steps, per episode
    assert split.overhead == pytest.approx(0.34)
    assert sum(vars(split).values()) == pytest.approx(1.0)


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


def test_two_attended_runs_in_one_sidecar_get_a_window_each(tmp_path):
    """A process names its sidecar, so two attended runs against one telemetry directory write one file. An
    attended rollout opens no ``eval.pass`` span, so its episodes are roots and the parent they share says
    nothing about which run wrote them. The run id does — and without it one window spans both runs and
    reports 1040 s of wall for 80 s of work."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, start, end, f'ep-{run}', attrs={ATTR_EPISODE_VIRTUAL_S: 20.0}, run_id=run)
            for run, (start, end) in (('rik-0', (0, 40)), ('rik-1', (1000, 1040)))
        ],
    )

    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    assert report.episodes == 2
    assert report.window is WallWindow.W_EPISODES
    assert report.wall_s == pytest.approx(80.0)
    assert report.wall_split.between_episodes == pytest.approx(0.0)


def test_telemetry_with_neither_a_pass_nor_an_episode_names_what_is_missing(tmp_path):
    """Spans that carry no closed pass and no episode leave nothing to reduce. The refusal must say so: a run
    killed before its first episode finished did record telemetry, so blaming a missing ``--timing`` sends the
    reader after a flag they set."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_RESET, 0, 5, 'reset0')])

    with pytest.raises(CommandFailed, match=f'no closed `{SPAN_EVAL_PASS}` span and no `{SPAN_EPISODE}` span') as exc:
        _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    assert '--timing' not in str(exc.value)
    assert exc.value.fault is Fault.USAGE


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


def test_jetson_samples_with_no_gpu_memory_give_a_mean_load_and_no_vram_peak(tmp_path):
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}', [_span(SPAN_EVAL_PASS, 0, 10, 'pass0')])
    stats = [
        {
            STAT_T_NS: t,
            STAT_GPU_COUNT: 1,
            STAT_GPUS: [{GPU_INDEX: 0, GPU_UTIL_PCT: util, GPU_MEM_USED_B: None, GPU_PROC_MEM_B: None}],
        }
        for t, util in ((1, 20.0), (2, 40.0))
    ]
    (telemetry_dir / f'{HARNESS_PROCESS}{STATS_SUFFIX}').write_text(''.join(json.dumps(s) + '\n' for s in stats))

    report = _build_report(_read_spans_dir(telemetry_dir), _read_stats_dir(telemetry_dir), policy_gpu=None)

    sim = report.gpu.sim
    assert sim is not None
    assert sim.mean_util_pct == pytest.approx(30.0)
    assert (sim.peak_vram_gb, sim.peak_proc_vram_gb) == (None, None)
    _render(report)


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
    with pytest.raises(CommandFailed, match='fb') as exc:
        _parse_dmon(log)
    assert exc.value.fault is Fault.USAGE


def test_a_tilde_run_dir_and_dmon_log_resolve_against_home(tmp_path, monkeypatch):
    """The documented eval quickstart writes to a `~`-relative directory, so the reduce reads one back."""
    (tmp_path / 'run').mkdir()
    _fixture(tmp_path / 'run' / TELEMETRY_SUBDIR)
    (tmp_path / 'dmon.log').write_text('# gpu    sm    fb\n#  Idx     %    MB\n    0    50  1024\n')
    monkeypatch.setenv('HOME', str(tmp_path))

    timing_report(run_dir='~/run', gpu_policy_log='~/dmon.log')

    summary = json.loads((tmp_path / 'run' / 'timing_summary.json').read_text())
    assert summary['episodes'] == 2
    assert summary['gpu']['policy']['peak_vram_gb'] == pytest.approx(1.0)


def test_a_remote_run_dir_reaches_pos3_verbatim(tmp_path, monkeypatch):
    """Expansion is the local branch's alone: an `s3://` URI is the remote store's to parse, `~` and all."""
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    downloaded, uploaded = [], []
    monkeypatch.setattr(pos3, 'download', lambda uri: downloaded.append(uri) or tmp_path)
    monkeypatch.setattr(pos3, 'upload', lambda key, path, delete=True: uploaded.append(key))

    timing_report(run_dir='s3://bucket/~run')

    assert downloaded == ['s3://bucket/~run']
    assert uploaded == ['s3://bucket/~run.timing_summary.json']


def test_a_run_dir_holding_no_telemetry_directory_names_it(tmp_path):
    with pytest.raises(CommandFailed, match='no telemetry directory') as excinfo:
        timing_report(run_dir=str(tmp_path))
    assert str(tmp_path / TELEMETRY_SUBDIR) in str(excinfo.value)
    assert excinfo.value.fault is Fault.USAGE


def test_a_telemetry_directory_holding_no_spans_is_not_reported_as_untimed(tmp_path):
    """The sidecar directory exists, so the run was timed — what is missing is its flushed spans."""
    (tmp_path / TELEMETRY_SUBDIR).mkdir()
    with pytest.raises(CommandFailed, match='carries no spans') as exc:
        timing_report(run_dir=str(tmp_path))
    assert exc.value.fault is Fault.USAGE


def test_positronic_exits_with_the_usage_status_for_a_directory_that_is_no_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['positronic', 'eval', 'timing-report', f'--run_dir={tmp_path}'])

    with pytest.raises(SystemExit) as exit_info:
        _internal_main()

    assert exit_info.value.code == Fault.USAGE
    assert 'no telemetry directory' in capsys.readouterr().err


def test_positronic_exits_with_the_usage_status_for_a_gpu_log_it_cannot_read(tmp_path, monkeypatch, capsys):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    missing = tmp_path / 'dmon.log'
    monkeypatch.setattr(
        'sys.argv', ['positronic', 'eval', 'timing-report', f'--run_dir={tmp_path}', f'--gpu_policy_log={missing}']
    )

    with pytest.raises(SystemExit) as exit_info:
        _internal_main()

    assert exit_info.value.code == Fault.USAGE
    assert str(missing) in capsys.readouterr().err


def test_a_gpu_log_that_is_not_text_is_a_usage_fault_naming_it(tmp_path):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    log = tmp_path / 'dmon.log'
    log.write_bytes(b'# gpu sm fb\n\xff\xfe\n')

    with pytest.raises(CommandFailed, match=re.escape(str(log))) as exc:
        timing_report(run_dir=str(tmp_path), gpu_policy_log=str(log))

    assert exc.value.fault is Fault.USAGE


@pytest.mark.parametrize('suffix', [SPANS_SUFFIX, STATS_SUFFIX], ids=['spans', 'stats'])
def test_a_sidecar_it_cannot_open_is_a_usage_fault_naming_it(tmp_path, suffix):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    unreadable = tmp_path / TELEMETRY_SUBDIR / f'unreadable{suffix}'
    unreadable.mkdir()

    with pytest.raises(CommandFailed, match=re.escape(str(unreadable))) as exc:
        timing_report(run_dir=str(tmp_path))

    assert exc.value.fault is Fault.USAGE


@pytest.mark.parametrize('argument', ['run_dir', 'gpu_policy_log'])
def test_a_home_directory_this_machine_does_not_have_is_a_usage_fault_naming_it(tmp_path, argument):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    paths = {'run_dir': str(tmp_path), 'gpu_policy_log': None, argument: '~no-such-user-on-this-box/x'}

    with pytest.raises(CommandFailed, match='~no-such-user-on-this-box/x') as exc:
        timing_report(**paths)

    assert exc.value.fault is Fault.USAGE


def test_a_summary_it_cannot_write_is_a_usage_fault_naming_it(tmp_path):
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    summary = tmp_path / 'timing_summary.json'
    summary.mkdir()

    with pytest.raises(CommandFailed, match=re.escape(str(summary))) as exc:
        timing_report(run_dir=str(tmp_path))

    assert exc.value.fault is Fault.USAGE


def _waypoint_fixture(telemetry_dir):
    """One pass, two episodes carrying waypoint accounts that differ in every field, so a sum, a share, an
    emission-weighted mean and a maximum are each told apart from the others."""
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
            _span(
                SPAN_EPISODE,
                0,
                40,
                'ep0',
                'pass0',
                {
                    ATTR_EPISODE_VIRTUAL_S: 20.0,
                    ATTR_WAYPOINTS_SCHEDULED: 100,
                    ATTR_WAYPOINTS_EMITTED: 10,
                    ATTR_WAYPOINTS_DROPPED: 30,
                    ATTR_WAYPOINTS_LATE_SUM_MS: 50.0,
                    ATTR_WAYPOINTS_LATE_MAX_MS: 20.0,
                },
            ),
            _span(
                SPAN_EPISODE,
                50,
                90,
                'ep1',
                'pass0',
                {
                    ATTR_EPISODE_VIRTUAL_S: 20.0,
                    ATTR_WAYPOINTS_SCHEDULED: 100,
                    ATTR_WAYPOINTS_EMITTED: 30,
                    ATTR_WAYPOINTS_DROPPED: 10,
                    ATTR_WAYPOINTS_LATE_SUM_MS: 30.0,
                    ATTR_WAYPOINTS_LATE_MAX_MS: 12.0,
                },
            ),
        ],
    )


def test_the_waypoint_account_sums_over_the_pass(tmp_path):
    """Counts add, the drop share is of what came due, the lateness mean is the pass's own emissions divided
    into the pass's own sum, and the maximum is the worse episode's."""
    _waypoint_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    assert report.waypoints is not None
    waypoints = report.waypoints
    assert waypoints.scheduled == 200
    assert waypoints.emitted == 40
    assert waypoints.dropped == 40
    assert waypoints.dropped_share == pytest.approx(0.5)  # 40 of the 80 that came due
    assert waypoints.mean_late_ms == pytest.approx(2.0)  # 80 ms over the pass's own 40 emissions
    assert waypoints.max_late_ms == pytest.approx(20.0)


def test_render_shows_the_waypoint_account(tmp_path):
    _waypoint_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    rendered = _render(report).splitlines()

    assert 'waypoints:           200 scheduled, 40 emitted, 40 dropped' in rendered
    assert 'waypoint drops:        50.0% of the waypoints that came due' in rendered
    assert 'waypoint late mean:  2.0 ms (max 20.0)' in rendered


def test_a_pass_whose_episodes_carry_no_waypoint_account_reports_none(tmp_path):
    """A run that scheduled no waypoint reduces to no waypoint block at all, rather than to a row of zeros
    that reads as a loop keeping perfect time."""
    _fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    assert report.waypoints is None
    assert not any(line.startswith('waypoint') for line in _render(report).splitlines())


def _one_waypoint_episode(telemetry_dir, *, emitted: int, dropped: int, late_sum_ms: float, late_max_ms: float):
    """One pass, one episode carrying exactly this waypoint account."""
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EVAL_PASS, 0, 100, 'pass0'),
            _span(
                SPAN_EPISODE,
                0,
                40,
                'ep0',
                'pass0',
                {
                    ATTR_EPISODE_VIRTUAL_S: 20.0,
                    ATTR_WAYPOINTS_SCHEDULED: 100,
                    ATTR_WAYPOINTS_EMITTED: emitted,
                    ATTR_WAYPOINTS_DROPPED: dropped,
                    ATTR_WAYPOINTS_LATE_SUM_MS: late_sum_ms,
                    ATTR_WAYPOINTS_LATE_MAX_MS: late_max_ms,
                },
            ),
        ],
    )


def test_a_pass_whose_waypoints_never_came_due_reports_no_rate(tmp_path):
    """An episode that schedules only waypoints ahead of it, and ends first, observes no rate at all. A zero
    reads as a loop that dropped nothing and ran on time."""
    _one_waypoint_episode(tmp_path / TELEMETRY_SUBDIR, emitted=0, dropped=0, late_sum_ms=0.0, late_max_ms=0.0)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    assert report.waypoints is not None
    assert report.waypoints.dropped_share is None
    assert report.waypoints.mean_late_ms is None
    assert report.waypoints.max_late_ms is None
    rendered = _render(report).splitlines()
    assert 'waypoint drops:      unavailable: no waypoint came due' in rendered
    assert 'waypoint late mean:  unavailable: no waypoint went out' in rendered


def test_a_pass_that_dropped_every_due_waypoint_reports_that_share(tmp_path):
    """The drop share and the lateness figures stand on their own populations. Every due waypoint dropped is
    a share of 100 per cent, and leaves no emission to be late."""
    _one_waypoint_episode(tmp_path / TELEMETRY_SUBDIR, emitted=0, dropped=30, late_sum_ms=0.0, late_max_ms=0.0)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    assert report.waypoints is not None
    assert report.waypoints.dropped_share == pytest.approx(1.0)
    assert report.waypoints.mean_late_ms is None
    assert report.waypoints.max_late_ms is None
    assert 'waypoint drops:       100.0% of the waypoints that came due' in _render(report).splitlines()


def test_an_episode_carrying_no_waypoint_account_is_left_out_and_said_so(tmp_path):
    """A directory mixing episodes that carry an account with episodes that do not reduces over the ones
    that carry one, and reports how many that was — an episode that measured nothing is not one that
    dropped nothing."""
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    _waypoint_fixture(telemetry_dir)
    _write_lines(
        telemetry_dir / f'older{SPANS_SUFFIX}',
        [
            _span(SPAN_EVAL_PASS, 200, 300, 'pass1'),
            _span(SPAN_EPISODE, 210, 250, 'ep2', 'pass1', {ATTR_EPISODE_VIRTUAL_S: 20.0}),
        ],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    assert report.episodes == 3
    assert report.waypoints is not None
    assert report.waypoints.episodes == 2
    assert report.waypoints.scheduled == 200  # the uninstrumented episode adds no zero of its own
    assert 'waypoints:           200 scheduled, 40 emitted, 40 dropped over 2 of 3 episodes' in _render(report)


_SERVED_MS = f'{ATTR_SERVED_PREFIX}{protocol.TIMING_SERVED}'
_SERVED_RECEIVED_NS = f'{ATTR_SERVED_PREFIX}{protocol.TIMING_RECEIVED_NS}'
_SERVED_SENT_NS = f'{ATTR_SERVED_PREFIX}{protocol.TIMING_SENT_NS}'
_SERVED_ENCODE_MS = f'{ATTR_SERVED_PREFIX}{protocol.TIMING_ENCODE}'
_NETWORK_ROW = f'wire.send + wire.recv - {_SERVED_MS} - {_SERVED_ENCODE_MS}'


def _call(prefix, start_s, served_ms, run_id='rig-run', sent_alone: bool | None = True, encode_ms: float | None = None):
    """One policy call on a worker, 0.1 s long: two overlapping image encodes, a prepare and a round trip.

    ``sent_alone`` ``None`` leaves the mark out, as a file from a client that does not write it.
    ``encode_ms`` ``None`` leaves out the encode of the answer's images, as a server that answers with no image.

    The parent of the `policy.submit` span is not in the file, as in a file that holds only the calls.
    """
    ms = 0.001
    submit, transform, infer = f'{prefix}-submit', f'{prefix}-transform', f'{prefix}-infer'
    return [
        _span(SPAN_POLICY_SUBMIT, start_s, start_s + 100 * ms, submit, f'{prefix}-step', run_id=run_id),
        _span('transform_images', start_s, start_s + 100 * ms, transform, submit, run_id=run_id),
        _span(
            SPAN_POLICY_ENCODE,
            start_s,
            start_s + 20 * ms,
            f'{prefix}-a',
            transform,
            {ATTR_CODEC: 'Jpeg'},
            run_id=run_id,
        ),
        _span(
            SPAN_POLICY_ENCODE,
            start_s + 10 * ms,
            start_s + 30 * ms,
            f'{prefix}-b',
            transform,
            {ATTR_CODEC: 'Jpeg'},
            run_id=run_id,
        ),
        _span(SPAN_POLICY_PREPARE, start_s + 30 * ms, start_s + 40 * ms, f'{prefix}-prepare', transform, run_id=run_id),
        _span(
            SPAN_POLICY_INFER,
            start_s + 40 * ms,
            start_s + 100 * ms,
            infer,
            transform,
            {_SERVED_MS: served_ms, _SERVED_RECEIVED_NS: 1e18, _SERVED_SENT_NS: 1e18}
            | ({} if sent_alone is None else {ATTR_WIRE_SENT_ALONE: sent_alone})
            | ({} if encode_ms is None else {_SERVED_ENCODE_MS: encode_ms}),
            run_id=run_id,
        ),
        _span(SPAN_WIRE_SEND, start_s + 45 * ms, start_s + 50 * ms, f'{prefix}-send', infer, run_id=run_id),
        _span(SPAN_WIRE_RECV, start_s + 50 * ms, start_s + 95 * ms, f'{prefix}-recv', infer, run_id=run_id),
    ]


def _latency_fixture(telemetry_dir):
    """Two episodes of one run with one call each, a call between them, and one timed step in the first."""
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, 0, 10, 'ep0', run_id='rig-run'),
            _span(SPAN_EPISODE, 20, 30, 'ep1', run_id='rig-run'),
            _span(
                SPAN_HARNESS_STEP,
                1,
                1.004,
                'step0',
                'ep0',
                {ATTR_STEP_LATE_MS: 0.5, ATTR_STEP_POLICY_MS: 2.0},
                run_id='rig-run',
            ),
            *_call('first', 2, served_ms=10.0),
            *_call('warm-up', 15, served_ms=500.0),
            *_call('second', 22, served_ms=30.0),
        ],
    )


def test_the_latency_of_each_segment_counts_the_spans_inside_the_episodes(tmp_path):
    _latency_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    rows = {row.segment: row for row in report.latency}
    assert rows[SPAN_POLICY_SUBMIT].count == 2  # the call between the episodes does not count
    assert rows[SPAN_POLICY_SUBMIT].episodes == 2
    assert rows[SPAN_POLICY_SUBMIT].p50_ms == pytest.approx(100.0)
    assert rows[SPAN_POLICY_INFER].p50_ms == pytest.approx(60.0)
    assert rows[SPAN_WIRE_RECV].p50_ms == pytest.approx(45.0)
    assert rows[_SERVED_MS].p50_ms == pytest.approx(20.0)
    assert rows[_SERVED_MS].p99_ms == pytest.approx(29.8)
    assert f'{SPAN_POLICY_ENCODE}[Jpeg]' in rows
    assert rows[f'{SPAN_POLICY_ENCODE}[Jpeg]'].count == 4
    assert rows[f'{SPAN_POLICY_ENCODE} per call'].p50_ms == pytest.approx(30.0)  # two encodes overlap by 10 ms
    assert rows[_NETWORK_ROW].p50_ms == pytest.approx(30.0)
    assert not any(segment.startswith(f'{SPAN_POLICY_INFER} -') for segment in rows)
    assert rows[SPAN_HARNESS_STEP].p50_ms == pytest.approx(4.0)
    assert rows[ATTR_STEP_LATE_MS].count == 1
    assert rows[ATTR_STEP_LATE_MS].episodes == 1
    assert rows[ATTR_STEP_POLICY_MS].p90_ms == pytest.approx(2.0)


def test_the_latency_leaves_out_the_server_instants_and_a_split_the_server_did_not_send(tmp_path):
    _latency_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    segments = {row.segment for row in report.latency}
    assert not segments & {_SERVED_RECEIVED_NS, _SERVED_SENT_NS, ATTR_WIRE_NETWORK_MS}


def _latency_rows_of(telemetry_dir, *spans):
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [_span(SPAN_EPISODE, 0, 10, 'ep0', run_id='rig-run'), *spans],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    return {row.segment: row for row in report.latency}


def test_the_latency_leaves_out_a_step_value_that_is_not_milliseconds(tmp_path):
    read_ms = 'step.read_ms.camera'
    step_attrs = {ATTR_STEP_GC_GENERATION: 2, ATTR_STEP_GC_MS: 3.0, read_ms: 4.0}
    rows = _latency_rows_of(
        tmp_path / TELEMETRY_SUBDIR, _span(SPAN_HARNESS_STEP, 1, 1.01, 'step0', 'ep0', step_attrs, run_id='rig-run')
    )

    assert ATTR_STEP_GC_GENERATION not in rows
    assert rows[ATTR_STEP_GC_MS].p50_ms == pytest.approx(3.0)
    assert rows[read_ms].p50_ms == pytest.approx(4.0)


def test_the_latency_leaves_out_the_one_way_wire_values_and_keeps_their_sum(tmp_path):
    wire_attrs = {ATTR_WIRE_UPLINK_MS: -5.0, ATTR_WIRE_DOWNLINK_MS: 25.0, ATTR_WIRE_NETWORK_MS: 20.0}
    rows = _latency_rows_of(
        tmp_path / TELEMETRY_SUBDIR, _span(SPAN_POLICY_INFER, 1, 1.1, 'infer', 'ep0', wire_attrs, run_id='rig-run')
    )

    assert ATTR_WIRE_UPLINK_MS not in rows
    assert ATTR_WIRE_DOWNLINK_MS not in rows
    assert rows[ATTR_WIRE_NETWORK_MS].p50_ms == pytest.approx(20.0)


def test_render_shows_the_latency_table(tmp_path):
    _latency_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    rendered = _render(report).splitlines()

    assert 'latency inside the episodes (ms):' in rendered
    assert any(re.fullmatch(r'  policy\.submit +2 +2 +100\.00 +100\.00 +100\.00', line) for line in rendered)


def test_a_report_with_no_timed_span_shows_no_latency_table(tmp_path):
    _waypoint_fixture(tmp_path / TELEMETRY_SUBDIR)
    report = _build_report(_read_spans_dir(tmp_path / TELEMETRY_SUBDIR), [], policy_gpu=None)

    assert report.latency == []
    assert not any(line.startswith('latency') for line in _render(report).splitlines())


def test_a_retried_call_gives_no_network_value(tmp_path):
    ms = 0.001
    telemetry_dir = tmp_path / TELEMETRY_SUBDIR
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [
            _span(SPAN_EPISODE, 0, 10, 'ep0', run_id='rig-run'),
            _span(
                SPAN_POLICY_INFER,
                1,
                1 + 100 * ms,
                'infer',
                'ep0',
                {_SERVED_MS: 20.0, ATTR_WIRE_SENT_ALONE: True},
                run_id='rig-run',
            ),
            # The first attempt loses its peer after the request is sent; the second attempt gets the answer.
            _span(SPAN_WIRE_SEND, 1 + 5 * ms, 1 + 10 * ms, 'send-1', 'infer', run_id='rig-run'),
            _span(SPAN_WIRE_RECV, 1 + 10 * ms, 1 + 30 * ms, 'recv-1', 'infer', run_id='rig-run'),
            _span(SPAN_WIRE_SEND, 1 + 40 * ms, 1 + 45 * ms, 'send-2', 'infer', run_id='rig-run'),
            _span(SPAN_WIRE_RECV, 1 + 45 * ms, 1 + 95 * ms, 'recv-2', 'infer', run_id='rig-run'),
        ],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)

    rows = {row.segment: row for row in report.latency}
    # The 10 ms reconnect between the attempts is not network time.
    assert _NETWORK_ROW not in rows
    assert rows['policy.infer'].count == 1


def _network_rows(telemetry_dir, *calls):
    telemetry_dir.mkdir()
    _write_lines(
        telemetry_dir / f'{HARNESS_PROCESS}{SPANS_SUFFIX}',
        [_span(SPAN_EPISODE, 0, 10, 'ep0', run_id='rig-run'), *(span for call in calls for span in call)],
    )
    report = _build_report(_read_spans_dir(telemetry_dir), [], policy_gpu=None)
    return [row for row in report.latency if row.segment == _NETWORK_ROW]


def test_the_network_row_counts_only_the_calls_sent_with_no_earlier_call_in_flight(tmp_path):
    [network] = _network_rows(
        tmp_path / TELEMETRY_SUBDIR,
        _call('alone', 1, served_ms=10.0),
        _call('queued', 2, served_ms=5.0, sent_alone=False),
    )
    assert network.count == 1
    assert network.p50_ms == pytest.approx(40.0)


def test_the_network_row_leaves_out_the_server_encode_of_the_answer_images(tmp_path):
    [network] = _network_rows(tmp_path / TELEMETRY_SUBDIR, _call('images', 1, served_ms=10.0, encode_ms=15.0))
    assert network.p50_ms == pytest.approx(25.0)  # 50 ms on the wire, 10 ms served, 15 ms encoding the answer


def test_a_file_whose_calls_carry_no_alone_mark_gives_no_network_row(tmp_path):
    rows = _network_rows(tmp_path / TELEMETRY_SUBDIR, _call('old', 1, served_ms=10.0, sent_alone=None))
    assert rows == []
