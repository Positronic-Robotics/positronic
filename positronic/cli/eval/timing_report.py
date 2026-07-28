"""Reduce the wall-clock telemetry sidecars written by ``positronic eval run --timing`` into a pass report.

Everything is read back from the per-process sidecar files under ``<run_dir>/telemetry/`` — the harness
process's nested spans (the pass, each episode, and its reset / env-step / materialize / record-IO / inference
phases) and its machine-load stats stream, plus the env server's own spans file (its step decomposed into
physics / render). A virtual-clock rollout cannot recover these wall costs from its own timestamps, so the
report is an offline reduce over the raw spans and samples: the per-phase wall split, the env-step split, the
inference-latency distribution, the real-time factor (recorded virtual duration over span wall) and the sim
box's GPU load. The policy endpoint (a different box) folds in from an optional ``nvidia-smi dmon`` log.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3

from positronic.telemetry import (
    ATTR_EPISODE_ABORTED,
    ATTR_EPISODE_PARTIAL,
    ATTR_EPISODE_VIRTUAL_S,
    ATTR_PASS_FAILED,
    HARNESS_PROCESS,
    SPAN_ENV_STEP,
    SPAN_EPISODE,
    SPAN_EVAL_PASS,
    SPAN_MATERIALIZE,
    SPAN_POLICY_INFER,
    SPAN_RECORD_IO,
    SPAN_RESET,
    TELEMETRY_SUBDIR,
    SpanRec,
    read_spans,
    read_stats,
)

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3


@dataclass
class GpuSummary:
    """Mean utilisation and peak VRAM for one box.

    ``peak_proc_vram_gb`` is the peak memory attributed to this eval's process tree; it is ``None`` for a
    policy ``dmon`` log, which carries no per-process attribution (only the whole box).
    """

    mean_util_pct: float
    peak_vram_gb: float
    peak_proc_vram_gb: float | None


@dataclass
class GpuReport:
    """GPU summaries per box: the sim box (from the recorded stats stream) and the policy endpoint.

    ``sim`` is ``None`` on a CPU sim box (no GPU samples); ``policy`` is ``None`` when no policy ``dmon`` log
    was passed in.
    """

    sim: GpuSummary | None
    policy: GpuSummary | None


@dataclass
class WallSplit:
    """Each phase's share of pass wall time.

    ``overhead`` is the within-episode wall unattributed to a measured phase (including the recorder's
    parquet/video close flush, which runs after record IO); ``between_episodes`` is the inter-episode wall
    (session teardown, homing, world rebuild) inside the pass span between one episode's finish and the next's
    start. The measured phases plus these two cover the pass span minus any aborted-episode wall, which is
    excluded from W_pass entirely.
    """

    reset: float
    env_step: float
    policy_wait: float
    record_io: float
    overhead: float
    between_episodes: float


@dataclass
class EnvStepSplit:
    """Fractions of the client-observed env-step wall, from the env server's own decomposition.

    ``phases`` maps each env-reported phase (physics, render, and the ``server_other`` residual — the env owns
    the phase set, so it is a map, not fixed fields) to its share of the client step wall; ``wire`` is the
    client step wall minus the server's whole in-step wall and the client materialisation (socket + codec);
    ``materialize`` is the client-side observation materialisation. ``None`` when no env server reported a
    decomposition (a native sim carries none).
    """

    phases: dict[str, float]
    wire: float
    materialize: float


@dataclass
class PassReport:
    """Pass-level wall-clock roll-up reduced from the recorded telemetry spans and stats."""

    episodes: int
    wall_pass_s: float
    real_time_factor: float
    policy_busy_fraction: float
    infer_calls: int
    infer_p50_ms: float
    infer_p95_ms: float
    wall_split: WallSplit
    env_step_split: EnvStepSplit | None
    gpu: GpuReport


def _dur_s(span: SpanRec) -> float:
    return (span.end_ns - span.start_ns) / 1e9


def _read_spans_dir(telemetry_dir: Path) -> list[SpanRec]:
    """Every span across all per-process ``*.spans.jsonl`` files under the telemetry dir."""
    spans: list[SpanRec] = []
    for path in sorted(telemetry_dir.glob('*.spans.jsonl')):
        spans.extend(read_spans(path))
    return spans


def _read_stats_dir(telemetry_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for path in sorted(telemetry_dir.glob('*.stats.jsonl')):
        samples.extend(read_stats(path))
    return samples


def _gpu_summary_from_stats(stats: list[dict], pass_windows: list[tuple[int, int]]) -> GpuSummary | None:
    """The sim box's GPU summary from the machine-load stream: mean utilisation over every per-GPU reading,
    and peak VRAM as the box-wide total — each sample's devices summed first, then the max over samples, so a
    multi-GPU box's peak is what the box held at one instant, not the largest single device. Only samples
    taken inside a completed pass's wall window count — a reused directory carries an earlier (possibly
    killed) run's samples, the stats twin of the orphan-episode exclusion. ``None`` when no counted sample
    carried a GPU (a CPU sim box)."""
    in_window = [sample for sample in stats if any(start <= int(sample['t_ns']) <= end for start, end in pass_windows)]
    utils: list[float] = []
    mem: list[float] = []
    proc: list[float] = []
    for sample in in_window:
        gpus = sample.get('gpus', [])
        utils.extend(float(gpu['util_pct']) for gpu in gpus)
        if gpus:
            mem.append(sum(float(gpu['mem_used_b']) for gpu in gpus))
        # Box-wide process VRAM needs every GPU attributed. A device whose NVML query errors mid-run is dropped
        # from the sample entirely, not left ``None``, so a sample is complete only when its present device
        # count equals the ``gpu_count`` the sampler recorded (its NVML handle count) AND every device reported
        # ``proc_mem_b``; an omitted device or a ``None`` reading each make the sum an undercount, so an
        # incomplete sample contributes nothing. If no sample is complete, ``proc`` stays empty and the peak
        # reads ``None``.
        if gpus and len(gpus) == sample['gpu_count'] and all(gpu.get('proc_mem_b') is not None for gpu in gpus):
            proc.append(sum(float(gpu['proc_mem_b']) for gpu in gpus))
    if not utils and not mem:
        return None
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else 0.0,
        peak_vram_gb=(max(mem) / _BYTES_PER_GB) if mem else 0.0,
        peak_proc_vram_gb=(max(proc) / _BYTES_PER_GB) if proc else None,
    )


def _parse_dmon(log_path: Path) -> GpuSummary:
    """Mean SM utilisation and peak framebuffer (GB) from a policy endpoint's ``nvidia-smi dmon`` log.

    Column layout varies — ``-o DT`` prepends date/time, ``-s u`` adds encoder/decoder (and, on newer
    drivers, JPEG/OFA) columns before the framebuffer — so the positions are read from the ``# ... sm ...
    fb ...`` name header rather than hard-coded. ``sm`` is SM utilisation (%) and ``fb`` the framebuffer
    use (MiB). Rows before the header, or with missing numeric fields, are skipped; a log whose header has
    ``sm`` but no ``fb`` (plain ``dmon`` / ``-s u``) fails loudly rather than silently reporting 0. A dmon
    log carries no per-process attribution, so ``peak_proc_vram_gb`` is ``None``.
    """
    sm_idx: int | None = None
    fb_idx: int | None = None
    gpu_idx: int | None = None
    utils: list[float] = []
    # dmon prints one row per device per sampling interval; peak VRAM is the box-wide total at one instant,
    # so rows group into cycles on the ``gpu`` index (a repeating index opens the next cycle) and the peak is
    # the max over per-cycle sums — matching the sim summary's per-sample device sum.
    cycle_fb: dict[str, float] = {}
    cycle_totals: list[float] = []

    def flush_cycle() -> None:
        if cycle_fb:
            cycle_totals.append(sum(cycle_fb.values()))
            cycle_fb.clear()

    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # The name header carries the column names ('sm', 'fb', ...); the units line ('%', 'MB') has no 'sm'.
            names = stripped.lstrip('#').split()
            if 'sm' in names:
                if 'fb' not in names:
                    # Plain ``nvidia-smi dmon`` (or ``-s u``) omits the framebuffer column, so every row would
                    # be skipped and peak VRAM silently read as 0. Fail loudly instead of under-reporting.
                    raise ValueError(
                        f'{log_path}: nvidia-smi dmon log has an `sm` column but no `fb` (framebuffer) column, so '
                        f'peak VRAM cannot be read and every sample would be dropped. Re-collect it with '
                        f'`nvidia-smi dmon -s um` (u=utilisation, m=memory), which emits the `fb` column.'
                    )
                sm_idx, fb_idx = names.index('sm'), names.index('fb')
                gpu_idx = names.index('gpu') if 'gpu' in names else None
            continue
        if sm_idx is None or fb_idx is None:
            continue
        row = line.split()
        try:
            sm = float(row[sm_idx])
            fb = float(row[fb_idx])
            device = row[gpu_idx] if gpu_idx is not None else '0'
        except (IndexError, ValueError):
            continue
        if device in cycle_fb:
            flush_cycle()
        cycle_fb[device] = fb
        utils.append(sm)
    flush_cycle()
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else 0.0,
        peak_vram_gb=(max(cycle_totals) / 1024.0) if cycle_totals else 0.0,
        peak_proc_vram_gb=None,
    )


@dataclass
class _EpisodeTiming:
    """One episode's wall aggregate reduced from its span subtree; fields are seconds unless named otherwise.
    ``env_step_s`` is the client env-step wall (materialisation included); ``materialize_s`` is that
    materialisation alone; ``overhead_s`` is the episode wall left unattributed after the measured phases."""

    wall_s: float
    virtual_s: float
    reset_s: float
    env_step_s: float
    materialize_s: float
    record_io_s: float
    policy_wait_s: float
    overhead_s: float
    infer_ms: list[float]


def _episode_timing(episode: SpanRec, children: dict[str, list[SpanRec]]) -> _EpisodeTiming:
    kids = children.get(episode.span_id, [])
    reset_s = sum(_dur_s(k) for k in kids if k.name == SPAN_RESET)
    env_step_s = sum(_dur_s(k) for k in kids if k.name == SPAN_ENV_STEP)
    record_io_s = sum(_dur_s(k) for k in kids if k.name == SPAN_RECORD_IO)
    infer_ms = [_dur_s(k) * 1000.0 for k in kids if k.name == SPAN_POLICY_INFER]
    materialize_s = sum(
        _dur_s(child)
        for k in kids
        if k.name == SPAN_ENV_STEP
        for child in children.get(k.span_id, [])
        if child.name == SPAN_MATERIALIZE
    )
    policy_wait_s = sum(infer_ms) / 1000.0
    wall_s = _dur_s(episode)
    measured = reset_s + env_step_s + record_io_s + policy_wait_s
    return _EpisodeTiming(
        wall_s=wall_s,
        virtual_s=float(episode.attrs.get(ATTR_EPISODE_VIRTUAL_S, 0.0)),
        reset_s=reset_s,
        env_step_s=env_step_s,
        materialize_s=materialize_s,
        record_io_s=record_io_s,
        policy_wait_s=policy_wait_s,
        overhead_s=max(wall_s - measured, 0.0),
        infer_ms=infer_ms,
    )


def _env_step_split(spans: list[SpanRec], episodes: list[SpanRec], env_step_sum: float, materialize_sum: float):
    """The env-step decomposition from the env server's own spans: a server ``env.step`` (recorded in the env
    process's own file, so ``process`` is not the harness) with physics/render children. Only server steps that
    start inside a completed episode's wall window count — an aborted rollout's steps would otherwise skew
    fractions whose denominator covers completed episodes only. ``None`` when no such span exists (a native sim
    reports no decomposition)."""

    def in_completed_episode(ts_ns: int) -> bool:
        return any(e.start_ns <= ts_ns <= e.end_ns for e in episodes)

    server_steps = [
        s
        for s in spans
        if s.name == SPAN_ENV_STEP and s.process != HARNESS_PROCESS and in_completed_episode(s.start_ns)
    ]
    server_step_sum = sum(_dur_s(s) for s in server_steps)
    if not server_step_sum or not env_step_sum:
        return None
    server_ids = {s.span_id for s in server_steps}
    phase_sums: dict[str, float] = defaultdict(float)
    for span in spans:
        if span.parent_id in server_ids:
            phase_sums[span.name] += _dur_s(span)
    phases = {name: total / env_step_sum for name, total in phase_sums.items()}
    server_other = max(server_step_sum - sum(phase_sums.values()), 0.0)
    if server_other:
        phases['server_other'] = server_other / env_step_sum
    return EnvStepSplit(
        phases=phases,
        wire=(env_step_sum - server_step_sum - materialize_sum) / env_step_sum,
        materialize=materialize_sum / env_step_sum,
    )


def _build_report(spans: list[SpanRec], stats: list[dict], policy_gpu: GpuSummary | None) -> PassReport:
    children: dict[str, list[SpanRec]] = defaultdict(list)
    for span in spans:
        if span.parent_id is not None:
            children[span.parent_id].append(span)

    # W_pass is the pass span's wall (summed if several passes appended to one dir), so inter-episode teardown
    # counts in the denominator; the wall gap between two separate passes falls outside every pass span. An
    # aborted episode's wall is subtracted out: the episode is dropped as invalid data (its virtual time, phases
    # and infers never reduce), so its wall must also leave every W_pass-normalised figure rather than land in
    # ``between_episodes`` and deflate the policy-busy / real-time factors.
    passes = [p for p in spans if p.name == SPAN_EVAL_PASS]
    pass_ids = {p.span_id for p in passes}
    aborted_wall = float(
        sum(
            _dur_s(s)
            for s in spans
            if s.name == SPAN_EPISODE and s.attrs.get(ATTR_EPISODE_ABORTED, False) and s.parent_id in pass_ids
        )
    )
    wall_pass = float(sum(_dur_s(p) for p in passes)) - aborted_wall
    # A failed pass still reduces — its partial window, episodes and samples are real recorded data — but
    # never silently: the mix is named so a skewed-looking split has its explanation on the console.
    failed = sum(bool(p.attrs.get(ATTR_PASS_FAILED, False)) for p in passes)
    if failed:
        logger.warning(
            '%d of %d pass(es) failed mid-run; their partial windows are included in the report', failed, len(passes)
        )
    # Only episodes under a completed pass reduce: a killed run flushes its episodes but never writes its
    # ``eval.pass`` span, and such orphans would inflate every pass-normalized figure when the directory is
    # reused for a later run.
    episodes = [s for s in spans if s.name == SPAN_EPISODE and not s.attrs.get(ATTR_EPISODE_ABORTED, False)]
    # A partial episode (its rollout failed mid-run) is kept, not dropped: its finished phases are real wall
    # and must attribute rather than fall into ``between_episodes``. Named, like ``pass.failed``, so the split
    # is read with its incomplete window in view.
    partial = sum(bool(e.attrs.get(ATTR_EPISODE_PARTIAL, False)) for e in episodes)
    if partial:
        logger.warning(
            '%d episode(s) did not run to completion (a failed pass?); their finished phases are included', partial
        )
    orphans = sum(e.parent_id not in pass_ids for e in episodes)
    if orphans:
        logger.warning('%d episode(s) belong to no completed pass (a killed run?); excluded from the report', orphans)
        episodes = [e for e in episodes if e.parent_id in pass_ids]
    timings = [_episode_timing(e, children) for e in episodes]

    episode_wall_sum = float(sum(t.wall_s for t in timings))
    env_step_sum = float(sum(t.env_step_s for t in timings))
    materialize_sum = float(sum(t.materialize_s for t in timings))
    all_infer_ms = np.array([ms for t in timings for ms in t.infer_ms], dtype=float)

    def phase_fraction(total: float) -> float:
        return (total / wall_pass) if wall_pass else 0.0

    policy_wait_sum = float(sum(t.policy_wait_s for t in timings))
    wall_split = WallSplit(
        reset=phase_fraction(sum(t.reset_s for t in timings)),
        env_step=phase_fraction(env_step_sum),
        policy_wait=phase_fraction(policy_wait_sum),
        record_io=phase_fraction(sum(t.record_io_s for t in timings)),
        overhead=phase_fraction(sum(t.overhead_s for t in timings)),
        between_episodes=phase_fraction(wall_pass - episode_wall_sum),
    )
    return PassReport(
        episodes=len(episodes),
        wall_pass_s=wall_pass,
        real_time_factor=(sum(t.virtual_s for t in timings) / wall_pass) if wall_pass else 0.0,
        policy_busy_fraction=phase_fraction(policy_wait_sum),
        infer_calls=int(all_infer_ms.size),
        infer_p50_ms=float(np.percentile(all_infer_ms, 50)) if all_infer_ms.size else 0.0,
        infer_p95_ms=float(np.percentile(all_infer_ms, 95)) if all_infer_ms.size else 0.0,
        wall_split=wall_split,
        env_step_split=_env_step_split(spans, episodes, env_step_sum, materialize_sum),
        gpu=GpuReport(sim=_gpu_summary_from_stats(stats, [(p.start_ns, p.end_ns) for p in passes]), policy=policy_gpu),
    )


def _render(report: PassReport) -> str:
    lines = [
        f'episodes:            {report.episodes}',
        f'W_pass (wall):       {report.wall_pass_s:.1f} s ({report.wall_pass_s / 3600:.2f} h)',
        f'real-time factor:    {report.real_time_factor:.3f} (sim-s per wall-s)',
        f'policy busy (k):     {report.policy_busy_fraction:.3f}  -> ~{1 / report.policy_busy_fraction:.1f} sims/H100'
        if report.policy_busy_fraction
        else 'policy busy (k):     n/a',
        f'infer calls:         {report.infer_calls}',
        f'infer p50 / p95:     {report.infer_p50_ms:.1f} / {report.infer_p95_ms:.1f} ms',
        'wall split (fraction of W_pass):',
    ]
    lines += [f'  {f.name:<12} {getattr(report.wall_split, f.name):.3f}' for f in fields(WallSplit)]
    if report.env_step_split is not None:
        split = report.env_step_split
        lines.append('env-step split (fraction of env_step):')
        lines += [f'  {name:<14} {frac:.3f}' for name, frac in split.phases.items()]
        lines.append(f'  {"wire":<14} {split.wire:.3f}')
        lines.append(f'  {"materialize":<14} {split.materialize:.3f}')
    for f in fields(GpuReport):
        summary = getattr(report.gpu, f.name)
        if summary is not None:
            line = f'gpu[{f.name}]: util {summary.mean_util_pct:.0f}%  peak VRAM {summary.peak_vram_gb:.1f} GB'
            if summary.peak_proc_vram_gb is not None:
                line += f'  (this eval {summary.peak_proc_vram_gb:.1f} GB)'
            lines.append(line)
    return '\n'.join(lines)


@cfn.config(gpu_policy_log=None)
def timing_report(run_dir: str, gpu_policy_log: str | None):
    """Reduce the telemetry sidecars under ``<run_dir>/telemetry/`` into a pass report.

    ``run_dir`` may be an ``s3://`` URI — the documented Nebius eval path writes there — in which case it is
    pulled local first, mirroring how the eval command syncs its output. The sim box's GPU load comes from the
    recorded machine-load stats stream; ``gpu_policy_log`` is an optional ``nvidia-smi dmon -s um`` log for the
    policy endpoint (a different box) folded in the same way. Writes ``timing_summary.json`` under the input —
    for an ``s3://`` input to the sibling key ``<run_dir>.timing_summary.json`` (pos3 forbids uploading inside
    the downloaded prefix) — and prints the report.
    """
    root = Path(pos3.download(run_dir)) if '://' in run_dir else Path(run_dir)
    telemetry_dir = root / TELEMETRY_SUBDIR
    spans = _read_spans_dir(telemetry_dir) if telemetry_dir.is_dir() else []
    if not any(s.name == SPAN_EVAL_PASS for s in spans):
        raise ValueError(f'no telemetry under {telemetry_dir} (recorded without --timing?)')

    policy_gpu = _parse_dmon(Path(gpu_policy_log)) if gpu_policy_log is not None else None
    report = _build_report(spans, _read_stats_dir(telemetry_dir), policy_gpu)
    summary_path = root / 'timing_summary.json'
    summary_path.write_text(json.dumps(asdict(report), indent=2))
    if '://' in run_dir:
        # A remote input was only downloaded, so the write above lands in the local cache. pos3 rejects an
        # upload inside a registered download prefix, so push the summary to a sibling key next to the run dir.
        pos3.upload(f'{run_dir.rstrip("/")}.timing_summary.json', summary_path, delete=False)
    logger.info(f'wrote {summary_path}')
    print(_render(report))
