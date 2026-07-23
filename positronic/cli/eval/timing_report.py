"""Reduce the per-rollout timing recorded by ``positronic eval run --timing`` into a pass-level report.

Everything is read back from the recorded episodes: the wall-clock costs a virtual-clock rollout cannot
recover from its own timestamps (the per-step ``timing.*`` signal + wall/finish/inference statics, summed
here per rollout), joined with the virtual duration (for the real-time factor), the on-disk size (bytes
per rollout) and the terminal flag. An optional ``nvidia-smi dmon`` log per box folds GPU utilisation and
peak VRAM into the same report.
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3

from positronic.dataset.episode import TELEMETRY_PREFIX, Episode
from positronic.dataset.local_dataset import load_all_datasets
from positronic.eval_timing import FINISHED_AT_KEY, GPU_LOG_FILENAME, INFER_MS_SIGNAL, RESET_S_KEY, WALL_S_KEY

logger = logging.getLogger(__name__)


@dataclass
class _RecordedFacts:
    """What the recorded episode contributes to the join, keyed by ``episode_uid``."""

    duration_s: float
    size_mb: float
    success: bool | None  # the env's eval.success verdict; None when it recorded none
    terminated: bool | None  # eval.terminated: the trial ended within budget vs timed out; None if unrecorded
    scored: bool  # eval.scored: the eval has a live success oracle, so a no-success termination is a failure


@dataclass
class _EpisodeTiming:
    """One rollout's wall-clock aggregate, recreated by summing its recorded ``timing.*`` signal and reading
    its statics — the per-episode reduce the recorder never stores. Fields are seconds unless named otherwise;
    ``env_step_s`` includes materialisation, ``env_client_s`` is materialisation alone. ``overhead_s`` is the
    wall unattributed to a measured phase. ``infer_ms`` is the raw per-call latency list."""

    episode_uid: str
    wall_s: float
    reset_s: float
    env_step_s: float
    policy_wait_s: float
    record_io_s: float
    overhead_s: float
    infer_ms: list[float]
    finished_at: float
    env_physics_s: float
    env_render_s: float
    env_server_s: float
    env_client_s: float


@dataclass
class GpuSummary:
    """Utilisation and peak framebuffer use parsed from one ``nvidia-smi dmon -s um`` log."""

    mean_util_pct: float
    peak_vram_gb: float


@dataclass
class GpuReport:
    """GPU summaries per box: the sim box (sampled into the dataset dir) and the policy endpoint box.

    Either is ``None`` when that box has no ``dmon`` log — a CPU sim box, or no policy log passed in.
    """

    sim: GpuSummary | None
    policy: GpuSummary | None


@dataclass
class WallSplit:
    """Each phase's share of pass wall time; all fractions sum to 1.

    ``overhead`` is the within-episode wall unattributed to a measured phase; ``between_episodes`` is the
    inter-episode wall (episode writer close — parquet/video flush — plus session teardown, homing, world
    rebuild) between one episode's finish and the next's start, which a per-episode sum drops. An episode
    seals its timing statics while its writer is open, so the writer-close flush lands here rather than in
    that episode's ``record_io``; the final episode's close falls outside the pass span entirely.
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

    ``wire`` is the client-side step wall minus the server's in-step wall and the client materialisation
    (socket + codec); ``materialize`` is the client-side observation materialisation (shared-memory image
    allocation + camera copies); ``server_other`` is the server's wall outside physics and rendering
    (managers, IK). ``None`` when no episode carried a server decomposition.
    """

    physics: float
    render: float
    server_other: float
    wire: float
    materialize: float


@dataclass
class PassReport:
    """Pass-level wall-clock roll-up reduced from the recorded episodes' timing signals, statics and facts."""

    episodes: int
    wall_pass_s: float
    real_time_factor: float
    policy_busy_fraction: float
    infer_calls: int
    infer_p50_ms: float
    infer_p95_ms: float
    wall_split: WallSplit
    env_step_split: EnvStepSplit | None
    mean_bytes_per_rollout: float
    success_rate: float | None  # None when no episode carried a success verdict (scored downstream)
    gpu: GpuReport


def _episode_timing(ep: Episode, uid: str) -> _EpisodeTiming | None:
    """Recreate one rollout's wall-clock aggregate from its recorded ``timing.*`` signals and statics.

    ``None`` when the episode carries no timing (recorded without ``--timing``). Each ``timing.<phase>_s``
    signal is the per-tick cost stream, summed here to the episode total; ``policy_wait_s`` is the sum of the
    per-call ``timing.infer_ms`` signal; ``reset_s`` and the wall scalars come from statics; ``overhead_s`` is
    the wall left unattributed after the measured phases.
    """
    if WALL_S_KEY not in ep.static:
        return None

    def phase_total(name: str) -> float:
        sig = ep.signals.get(f'{TELEMETRY_PREFIX}{name}')
        return float(sum(np.asarray(value).sum() for value, _ in sig)) if sig is not None else 0.0

    infer_sig = ep.signals.get(INFER_MS_SIGNAL)
    infer_ms = [float(value) for value, _ in infer_sig] if infer_sig is not None else []
    policy_wait_s = sum(infer_ms) / 1000.0
    reset_s = float(ep.static[RESET_S_KEY])
    env_step_s = phase_total('env_step_s')
    record_io_s = phase_total('record_io_s')
    wall_s = float(ep.static[WALL_S_KEY])
    measured = reset_s + env_step_s + policy_wait_s + record_io_s
    return _EpisodeTiming(
        episode_uid=uid,
        wall_s=wall_s,
        reset_s=reset_s,
        env_step_s=env_step_s,
        policy_wait_s=policy_wait_s,
        record_io_s=record_io_s,
        overhead_s=max(wall_s - measured, 0.0),
        infer_ms=infer_ms,
        finished_at=float(ep.static[FINISHED_AT_KEY]),
        env_physics_s=phase_total('env_physics_s'),
        env_render_s=phase_total('env_render_s'),
        env_server_s=phase_total('env_server_s'),
        env_client_s=phase_total('env_client_s'),
    )


def _load_episodes(dataset_dir: Path) -> tuple[list[_EpisodeTiming], dict[str, _RecordedFacts]]:
    """One pass over the recorded episodes: the per-rollout timing records (only those carrying timing) and
    the recorded facts (duration/size/success) every episode contributes to the join, keyed by uid."""
    dataset = load_all_datasets(dataset_dir)
    records: list[_EpisodeTiming] = []
    facts: dict[str, _RecordedFacts] = {}
    for i in range(len(dataset)):
        ep = dataset[i]
        uid = str(ep.meta.get('uid', f'ts-{ep.meta.get("created_ts_ns", i)}'))
        # ``eval.success`` is the env's task-success verdict; ``eval.terminated`` only means the episode
        # ended within budget, so a failed-but-done rollout is terminated=True, success=False and a
        # timed-out one is terminated=False with no success verdict. ``eval.scored`` says whether the eval
        # scores success at all, which is what makes a no-success termination a failure rather than unscored.
        verdict = ep.static.get('eval.success')
        terminated = ep.static.get('eval.terminated')
        facts[uid] = _RecordedFacts(
            duration_s=ep.duration_ns / 1e9,
            size_mb=float(ep.meta.get('size_mb', 0.0)),
            success=None if verdict is None else bool(verdict),
            terminated=None if terminated is None else bool(terminated),
            scored=bool(ep.static.get('eval.scored', False)),
        )
        timing = _episode_timing(ep, uid)
        if timing is not None:
            records.append(timing)
    return records, facts


def _parse_dmon(log_path: Path) -> GpuSummary:
    """Mean SM utilisation and peak framebuffer (GB) from an ``nvidia-smi dmon`` log.

    Column layout varies — ``-o DT`` prepends date/time, ``-s u`` adds encoder/decoder (and, on newer
    drivers, JPEG/OFA) columns before the framebuffer — so the positions are read from the ``# ... sm ...
    fb ...`` name header rather than hard-coded. ``sm`` is SM utilisation (%) and ``fb`` the framebuffer
    use (MiB). Rows before the header, or with missing numeric fields, are skipped rather than failing.
    """
    sm_idx: int | None = None
    fb_idx: int | None = None
    utils: list[float] = []
    fb_mib: list[float] = []
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # The name header carries both 'sm' and 'fb'; the units line ('%', 'MB') does not.
            names = stripped.lstrip('#').split()
            if 'sm' in names and 'fb' in names:
                sm_idx, fb_idx = names.index('sm'), names.index('fb')
            continue
        if sm_idx is None or fb_idx is None:
            continue
        fields = line.split()
        try:
            utils.append(float(fields[sm_idx]))
            fb_mib.append(float(fields[fb_idx]))
        except (IndexError, ValueError):
            continue
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else 0.0, peak_vram_gb=(max(fb_mib) / 1024.0) if fb_mib else 0.0
    )


def _success_outcome(facts: _RecordedFacts) -> bool | None:
    """Whether the episode is a success (True), a failure (False), or unscored (None).

    A recorded ``eval.success`` is taken as-is. With none, an episode of a *scored* eval that reached a
    verdict or timed out (``eval.terminated`` recorded) is a failure — this is the timed-out rollout an
    adapter only marks on live success. An episode of an unscored eval (success computed downstream) is not
    scored here at all.
    """
    if facts.success is not None:
        return facts.success
    if facts.scored and facts.terminated is not None:
        return False
    return None


def _build_report(records: list[_EpisodeTiming], facts: dict[str, _RecordedFacts], gpu: GpuReport) -> PassReport:
    # ``records`` and ``facts`` come from the same pass over the dataset, so every record's uid is a facts
    # key by construction: the episode count and wall time come from ``records``, RTF/bytes/success from facts.

    # W_pass is the true pass span — the first episode's start to the last's finish — so inter-episode
    # teardown wall (session close, homing, world rebuild) counts in the denominator. A per-episode sum drops
    # it, inflating the real-time factor and sizing numbers. ``finished_at - wall_s`` is each episode's start.
    episode_wall_sum = float(sum(r.wall_s for r in records))
    wall_pass = float(max(r.finished_at for r in records) - min(r.finished_at - r.wall_s for r in records))
    all_infer_ms = np.array([ms for r in records for ms in r.infer_ms], dtype=float)

    # Aggregate fraction = summed phase over W_pass, so long episodes weigh proportionally.
    def phase_fraction(attr: str) -> float:
        return float(sum(getattr(r, attr) for r in records) / wall_pass) if wall_pass else 0.0

    # The split covers only the episodes whose env reports its own decomposition: in a sweep mixing a
    # decomposing env server with a sim that reports none, the undecomposed episodes' env-step wall would
    # otherwise all land in ``wire``.
    decomposed = [r for r in records if r.env_server_s > 0]
    env_step_sum = float(sum(r.env_step_s for r in decomposed))
    env_step_split = None
    if env_step_sum:
        server_sum = float(sum(r.env_server_s for r in decomposed))
        physics_sum = float(sum(r.env_physics_s for r in decomposed))
        render_sum = float(sum(r.env_render_s for r in decomposed))
        client_sum = float(sum(r.env_client_s for r in decomposed))
        server_other = (server_sum - physics_sum - render_sum) / env_step_sum
        # Isaac Lab steps physics with `sim.step(render=False)` and renders once outside the decimation loop,
        # so the physics and render spans are disjoint and this stays non-negative — but a task config that
        # renders inside the physics loop would double-count that wall in both spans. Warn rather than clamp,
        # so a double count surfaces instead of hiding in a normalized split.
        if server_other < 0:
            logger.warning(
                f'server_other is negative ({server_other:.4f}): the env server double-counted wall between '
                f'physics and render (a render inside the physics loop?); the env_step_split is not trustworthy'
            )
        env_step_split = EnvStepSplit(
            physics=physics_sum / env_step_sum,
            render=render_sum / env_step_sum,
            server_other=server_other,
            wire=(env_step_sum - server_sum - client_sum) / env_step_sum,
            materialize=client_sum / env_step_sum,
        )

    matched = [facts[r.episode_uid] for r in records]
    sim_seconds = sum(f.duration_s for f in matched)
    # Each episode carries whether its eval scores success (``eval.scored``), so scoring needs no pass-wide
    # oracle: a scored eval's timeouts count as failures (even an all-timeout eval reads 0%), while an
    # unscored eval's episodes stay out of the rate entirely regardless of what else ran in the sweep.
    judged = [outcome for outcome in (_success_outcome(f) for f in matched) if outcome is not None]
    return PassReport(
        episodes=len(records),
        wall_pass_s=wall_pass,
        real_time_factor=(sim_seconds / wall_pass) if wall_pass else 0.0,
        policy_busy_fraction=phase_fraction('policy_wait_s'),
        infer_calls=int(all_infer_ms.size),
        infer_p50_ms=float(np.percentile(all_infer_ms, 50)) if all_infer_ms.size else 0.0,
        infer_p95_ms=float(np.percentile(all_infer_ms, 95)) if all_infer_ms.size else 0.0,
        wall_split=WallSplit(
            reset=phase_fraction('reset_s'),
            env_step=phase_fraction('env_step_s'),
            policy_wait=phase_fraction('policy_wait_s'),
            record_io=phase_fraction('record_io_s'),
            overhead=phase_fraction('overhead_s'),
            between_episodes=float((wall_pass - episode_wall_sum) / wall_pass) if wall_pass else 0.0,
        ),
        env_step_split=env_step_split,
        mean_bytes_per_rollout=(sum(f.size_mb for f in matched) / len(matched) * 1024 * 1024) if matched else 0.0,
        success_rate=(sum(judged) / len(judged)) if judged else None,
        gpu=gpu,
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
        f'bytes / rollout:     {report.mean_bytes_per_rollout / 1e6:.1f} MB',
        f'success rate:        {report.success_rate:.3f}'
        if report.success_rate is not None
        else 'success rate:        n/a',
        'wall split (fraction of W_pass):',
    ]
    lines += [f'  {f.name:<12} {getattr(report.wall_split, f.name):.3f}' for f in fields(WallSplit)]
    if report.env_step_split is not None:
        lines.append('env-step split (fraction of env_step):')
        lines += [f'  {f.name:<12} {getattr(report.env_step_split, f.name):.3f}' for f in fields(EnvStepSplit)]
    for f in fields(GpuReport):
        summary = getattr(report.gpu, f.name)
        if summary is not None:
            lines.append(f'gpu[{f.name}]: util {summary.mean_util_pct:.0f}%  peak VRAM {summary.peak_vram_gb:.1f} GB')
    return '\n'.join(lines)


@cfn.config(gpu_sim_log=None, gpu_policy_log=None)
def timing_report(dataset_dir: str, gpu_sim_log: str | None, gpu_policy_log: str | None):
    """Reduce the recorded per-rollout timing (the ``timing.*`` signal + statics on each episode) into a pass report.

    ``dataset_dir`` may be an ``s3://`` URI — the documented Nebius eval path writes there — in which case
    it is pulled local first, mirroring how the eval command syncs its output. ``gpu_sim_log`` /
    ``gpu_policy_log`` are optional ``nvidia-smi dmon`` logs (sim box / policy endpoint) whose utilisation
    and peak VRAM are folded in. Writes ``timing_summary.json`` next to the input — for an ``s3://`` input
    to the sibling key ``<dataset_dir>.timing_summary.json`` (pos3 forbids uploading inside the downloaded
    prefix) — and prints the report.
    """
    # Pull a remote dataset local before reading; a plain local path is used as-is (never handed to pos3,
    # whose download would prune it).
    root = Path(pos3.download(dataset_dir)) if '://' in dataset_dir else Path(dataset_dir)
    records, facts = _load_episodes(root)
    if not records:
        raise ValueError(f'no timed episodes under {root} (recorded without --timing?)')

    # The sim box samples its own GPU into the dataset dir under ``--timing``; an explicit path overrides
    # it, and the policy endpoint's log (a different box) is only ever passed in.
    sim_log = Path(gpu_sim_log) if gpu_sim_log is not None else root / GPU_LOG_FILENAME
    gpu = GpuReport(
        sim=_parse_dmon(sim_log) if sim_log.exists() else None,
        policy=_parse_dmon(Path(gpu_policy_log)) if gpu_policy_log is not None else None,
    )

    report = _build_report(records, facts, gpu)
    summary_path = root / 'timing_summary.json'
    summary_path.write_text(json.dumps(asdict(report), indent=2))
    if '://' in dataset_dir:
        # A remote input was only downloaded, so the write above lands in the local cache. pos3 rejects an
        # upload inside a registered download prefix, so push the summary to a sibling key next to the
        # dataset dir (the CLI's @pos3.with_mirror flushes it on exit; delete=False never prunes the bucket).
        pos3.upload(f'{dataset_dir.rstrip("/")}.timing_summary.json', summary_path, delete=False)
    logger.info(f'wrote {summary_path}')
    print(_render(report))
